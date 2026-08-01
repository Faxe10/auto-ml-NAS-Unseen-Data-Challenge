import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# unser import stuff
import random
import torch
from pyparsing import results
from tqdm.auto import tqdm
import copy
import json
import csv
import math
import time

# ======================================================================================================================
# GLOBAL CONFIGURATION
# ======================================================================================================================


torch.set_num_threads(22)  # = Anzahl physischer Kerne, nicht logischer
torch.set_num_interop_threads(2)  # niedrig halten, meist reicht 1-2

OPERATION_NAMES = ["conv3x3", "conv5x5", "maxpool3x3", "avgpool3x3", "skip", "none", "dil_sep_conv3x3"]
# standard: "conv3x3", "conv5x5", "maxpool3x3", "avgpool3x3", "skip", "none"
# new: "sep_conv3x3", "sep_conv5x5", "dil_sep_conv3x3", "conv1x1", "factorized_conv5x5", "avgpool3x3_conv1x1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_NODES = 5
NUM_CELLS = 2
EDGES = [(i, j) for j in range(1, NUM_NODES) for i in range(j)]
SEED = 42


# ======================================================================================================================
# NAS SEARCH WRAPPER
# ======================================================================================================================

class NAS:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The NAS class will receive the following inputs
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor

        You can modify or add anything into the metadata that you wish,
        if you want to pass messages between your classes,
    """

    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

    # --------------------------------------------------
    # MAIN ENTRY POINT
    # --------------------------------------------------

    def search(self):
        total_time = self.clock.check()
        search_time_limit = total_time * 0.80
        train_time_limit = total_time * 0.15

        print(
            f"⏱️ Total Time: {show_time(total_time)} | Search Budget: 80% ({show_time(search_time_limit)}) | Train Budget: 15% ({show_time(train_time_limit)})")

        self.metadata['target_training_time_seconds'] = train_time_limit

        time_per_epoch = self._estimate_epoch_time()
        print(f"⏱️ Estimated time per epoch: {time_per_epoch:.2f} seconds")

        eta = 3
        conservative_s_max = 3
        estimated_max_budget = int(
            (search_time_limit / time_per_epoch) / ((conservative_s_max + 1) * (eta / (eta - 1))))
        max_budget = max(5, min(estimated_max_budget, 50))

        print(f"⚙️ Dynamically set Hyperband max_budget_per_model to: {max_budget} epochs")

        # --- THE CONTINUOUS SEARCH LOOP ---
        global_search_start = time.time()

        ultimate_best_model_state = None
        ultimate_best_arch = None
        ultimate_best_val_acc = -1.0
        all_passes_results = []

        pass_num = 1

        # Keep spinning up new Hyperband passes until the 80% time runs out
        rng = random.Random(SEED)
        while (time.time() - global_search_start) < search_time_limit:

            elapsed_so_far = time.time() - global_search_start
            remaining_search_time = search_time_limit - elapsed_so_far

            # If we have less than a minute left in the search budget, don't start a whole new pass
            if remaining_search_time < 60:
                print(f"⏱️ Only {remaining_search_time:.0f}s left in search budget. Moving on to training phase.")
                break

            print(f"\n" + "=" * 60)
            print(
                f"🚀 Starting Hyperband Pass {pass_num} | Time remaining for Search: {show_time(remaining_search_time)}")
            print("=" * 60)

            # Pass the REMAINING time to this specific run
            best_model, results, best_arch, best_val_acc = self.hyperband_search(
                rng,
                min_budget_per_model=1,
                max_budget_per_model=max_budget,
                eta=eta,
                search_time_limit=remaining_search_time
            )

            all_passes_results.extend(results)

            # Keep track of the all-time best across all Hyperband loops
            if best_val_acc > ultimate_best_val_acc:
                ultimate_best_val_acc = best_val_acc
                ultimate_best_arch = best_arch
                ultimate_best_model_state = copy.deepcopy(best_model.state_dict())
                print(f"🏆 ALL-TIME BEST UPDATED: {ultimate_best_val_acc * 100:.2f}%")

            pass_num += 1

        print(f"\n🛑 Search Phase Complete! Executed {pass_num - 1} full Hyperband passes.")

        # --- RECONSTRUCT THE ULTIMATE BEST MODEL ---
        final_model = None
        if ultimate_best_arch is not None and ultimate_best_model_state is not None:
            in_channels = self.metadata['input_shape'][1]
            num_classes = self.metadata['num_classes']
            final_model = NASNetwork(ultimate_best_arch, NUM_CELLS, channels=16,
                                     num_classes=num_classes, in_channels=in_channels).to(DEVICE)
            final_model.load_state_dict(ultimate_best_model_state)

        return final_model

    def _estimate_epoch_time(self):
        """Runs a tiny fraction of an epoch to estimate the speed of the current dataset/hardware."""
        arch = self.random_architecture()
        model = self._build_model_from_arch(arch).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        start_time = time.time()
        max_batches_to_test = 10
        batches_run = 0

        for data, target in self.train_loader:
            if batches_run >= max_batches_to_test: break
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            batches_run += 1

        time_taken = time.time() - start_time
        total_batches = len(self.train_loader)

        # Free up VRAM
        del model
        del optimizer
        torch.cuda.empty_cache()

        # Extrapolate to full epoch
        return (time_taken / max(1, batches_run)) * total_batches

    def random_architecture(self, rng=None):
        """Sample a random architecture (dict mapping edge -> operation name)."""
        if rng is None:
            rng = random.Random()
        return {e: rng.choice(OPERATION_NAMES) for e in EDGES}

    # --------------------------------------------------
    # ZERO-SHOT PROXY - ACTIVE
    # --------------------------------------------------

    def _build_model_from_arch(self, arch, channels=16, num_cells=NUM_CELLS, device=DEVICE):
        """
        Build exactly the same model type that is later used in train_and_evaluate().
        This avoids scoring a different architecture than the one we actually train.
        """
        in_channels = self.metadata['input_shape'][1]
        num_classes = self.metadata['num_classes']

        return NASNetwork(
            arch,
            num_cells=NUM_CELLS,
            channels=channels,
            num_classes=num_classes,
            in_channels=in_channels
        ).to(device)

    def _get_zero_shot_batch(self, batch_size=16, device=DEVICE):
        """
        Take one small real batch from the training data.
        NASWOT needs real data, SynFlow does not.
        """
        data, target = next(iter(self.train_loader))
        data = data[:batch_size].to(device)
        target = target[:batch_size].to(device)
        return data, target

    def estimate_flops(self, model, x):
        """
        Lightweight FLOP estimate for Conv2d and Linear layers.
        This is not a perfect profiler, but good enough for hard filtering/ranking.
        """
        flops = 0
        hooks = []

        def conv_hook(module, inputs, output):
            nonlocal flops
            # output shape: [batch, out_channels, out_h, out_w]
            batch_size = output.shape[0]
            out_channels = output.shape[1]
            out_h = output.shape[2]
            out_w = output.shape[3]

            kernel_h, kernel_w = module.kernel_size
            in_channels = module.in_channels
            groups = module.groups

            kernel_ops = kernel_h * kernel_w * (in_channels // groups)
            flops += batch_size * out_channels * out_h * out_w * kernel_ops

        def linear_hook(module, inputs, output):
            nonlocal flops
            # output shape: [batch, out_features]
            flops += output.numel() * module.in_features

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(linear_hook))

        was_training = model.training
        model.eval()

        with torch.no_grad():
            _ = model(x)

        model.train(was_training)

        for hook in hooks:
            hook.remove()

        return int(flops)

    def hard_filter_architecture(
            self,
            arch,
            model=None,
            x=None,
            channels=16,
            num_cells=NUM_CELLS,
            device=DEVICE,
            max_params=None,
            max_flops=None,
            min_active_edges=2,
            max_none_edges=4,
            min_conv_edges=1,
    ):
        """
        Hard Filtering:
        - removes architectures with too many inactive edges
        - removes architectures without enough convolutional operations
        - removes too-large models by parameter count
        - removes too-slow models by approximate FLOPs
        - removes architectures whose forward pass fails or creates invalid outputs

        Returns:
            (is_valid, info_dict)
        """
        info = {
            "reason": "ok",
            "n_params": None,
            "flops": None,
            "active_edges": None,
            "none_edges": None,
            "conv_edges": None,
        }

        # TODO: Werte, nochmal checken
        if max_params is None:
            max_params = self.metadata.get("max_params", 100_000_000_000)
        if max_flops is None:
            max_flops = self.metadata.get("max_flops", 1_000_000_000_000)

        active_edges = sum(op != "none" for op in arch.values())
        none_edges = sum(op == "none" for op in arch.values())
        conv_edges = sum(op in ["conv3x3", "conv5x5"] for op in arch.values())

        info["active_edges"] = active_edges
        info["none_edges"] = none_edges
        info["conv_edges"] = conv_edges

        if active_edges < min_active_edges:
            info["reason"] = "too_few_active_edges"
            return False, info

        if none_edges > max_none_edges:
            info["reason"] = "too_many_none_edges"
            return False, info

        if conv_edges < min_conv_edges:
            info["reason"] = "too_few_conv_edges"
            return False, info

        owns_model = model is None
        try:
            if model is None:
                model = self._build_model_from_arch(
                    arch,
                    channels=channels,
                    num_cells=NUM_CELLS,
                    device=device
                )

            if x is None:
                x, _ = self._get_zero_shot_batch(batch_size=8, device=device)

            n_params = count_parameters(model)
            info["n_params"] = n_params

            if max_params is not None and n_params > max_params:
                info["reason"] = "too_many_parameters"
                return False, info

            model.eval()
            with torch.no_grad():
                output = model(x)

            if output.ndim != 2:
                info["reason"] = "invalid_output_shape"
                return False, info

            if output.shape[0] != x.shape[0]:
                info["reason"] = "invalid_batch_dimension"
                return False, info

            if not torch.isfinite(output).all():
                info["reason"] = "non_finite_forward_output"
                return False, info

            flops = self.estimate_flops(model, x)
            info["flops"] = flops

            if max_flops is not None and flops > max_flops:
                info["reason"] = "too_many_flops"
                return False, info

            return True, info

        except RuntimeError as e:
            #if "out of memory" in str(e).lower() and torch.cuda.is_available():
                #torch.cuda.empty_cache()
            info["reason"] = f"runtime_error: {str(e)[:120]}"
            return False, info

        except Exception as e:
            info["reason"] = f"error: {str(e)[:120]}"
            return False, info

        finally:
            if owns_model:
                del model
                #if torch.cuda.is_available():
                    #torch.cuda.empty_cache()

    def naswot_score(self, model, x):
        """
        NASWOT as the primary zero-shot ranking proxy because it estimates architecture quality from
        activation patterns at initialization without requiring gradient-based training

        NASWOT score:
        Uses the binary activation pattern after ReLU layers.
        Higher score usually means more expressive activation diversity.
        """
        model.eval()
        binary_activations = []
        hooks = []

        # Hook wird automatisch ausgeführt, sobald die ReLU Schicht aufgerufen wird
        def relu_hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            if output.ndim < 2:
                return

            activation = (output.detach() > 0).flatten(start_dim=1).float()
            binary_activations.append(activation)  # ReLU-Output wird in ein binäres Muster umgewandelt

        for module in model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(relu_hook))

        try:  # Forward-Pass
            with torch.no_grad():  # keine Gradienten werden gespeichert
                _ = model(x)

            if len(binary_activations) == 0:
                return float("-inf")

            batch_size = x.shape[0]
            kernel_matrix = torch.zeros(
                (batch_size, batch_size),
                device=x.device,
                dtype=torch.float32
            )

            # Für jede ReLU-Schicht wird gezählt, wie ähnlich die binären Aktivierungsmuster zweier Samples sind
            for activation in binary_activations:
                kernel_matrix += activation @ activation.t()  # beide Samples aktivieren denselben Neuron-Ausgang
                kernel_matrix += (1.0 - activation) @ (
                        1.0 - activation).t()  # beide Samples deaktivieren denselben Neuron-Ausgang
            # Ergebnis: Wie ähnlich reagieren zwei Eingabebilder auf die Architektur?

            # Numerical stability for log determinant
            eps = 1e-6
            kernel_matrix += eps * torch.eye(batch_size, device=x.device)

            sign, logdet = torch.linalg.slogdet(kernel_matrix)  # NASWOT-Score

            if sign <= 0 or not torch.isfinite(logdet):
                return float("-inf")

            return float(logdet.item())

        except RuntimeError:
            return float("-inf")

        finally:
            for hook in hooks:
                hook.remove()

    def synflow_score(self, model, input_shape=None, device=DEVICE):
        """
        SynFlow as an auxiliary trainability proxy to detect architectures with
        poor signal propagation at initialization

        SynFlow score:
        Data-independent trainability score.
        Linearizes the network by taking absolute weights, forwards an all-ones input,
        backpropagates the summed output, and sums |weight * gradient|.
        """
        if input_shape is None:
            input_shape = self.metadata["input_shape"]

        # metadata input_shape is expected as [n_samples, channels, height, width]
        channels = int(input_shape[1])
        height = int(input_shape[2])
        width = int(input_shape[3])

        signs = {}
        was_training = model.training

        try:
            model.eval()
            model.zero_grad(set_to_none=True)

            # Linearize model: Alle Parameter werden positiv
            # Damit sich positive und negative Pfade nicht gegenseitig auslöschen
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        signs[name] = torch.sign(parameter)
                        parameter.abs_()

            x = torch.ones((1, channels, height, width),
                           device=device)  # künstlicher Input, Implementierung datenunabhängig
            output = model(x)  # Forward-Pass
            torch.sum(output).backward()  # Backward-Pass

            score = 0.0
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    score += torch.sum(torch.abs(parameter * parameter.grad)).item()

            if not math.isfinite(score):
                return float("-inf")

            return float(score)

        except RuntimeError:
            return float("-inf")

        finally:
            # Restore original parameter signs.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and name in signs:
                        parameter.mul_(signs[name])

            model.zero_grad(set_to_none=True)
            model.train(was_training)

    def gradnorm_score(self, model, x, target):
        """
        GradNorm:
        Computes the sum of absolute gradients after one supervised loss backward pass.

        Interpretation:
        Higher GradNorm means the architecture produces stronger gradient signals at initialization.
        Very small values can indicate weak signal propagation or dead paths.
        """
        was_training = model.training

        try:
            model.eval()
            model.zero_grad(set_to_none=True)

            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()

            score = 0.0
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    score += torch.sum(torch.abs(parameter.grad)).item()

            if not math.isfinite(score):
                return float("-inf")

            return float(score)

        except RuntimeError:
            return float("-inf")

        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

    def snip_score(self, model, x, target):
        """
        SNIP:
        Computes one-shot connection sensitivity at initialization.

        Score:
            sum(|parameter * gradient|)

        Interpretation:
        Higher SNIP means the current architecture has weights that strongly affect
        the supervised loss on a small batch.
        """
        was_training = model.training

        try:
            model.eval()
            model.zero_grad(set_to_none=True)

            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()

            score = 0.0
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    score += torch.sum(torch.abs(parameter * parameter.grad)).item()

            if not math.isfinite(score):
                return float("-inf")

            return float(score)

        except RuntimeError:
            return float("-inf")

        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

    def zen_score(self, model, x, epsilon=1e-2):
        """
        Zen-Score inspired proxy:
        Measures how strongly the architecture changes its internal representation when
        the input is slightly perturbed.

        Practical implementation for this search space:
        - run x and x + noise through the initialized network
        - compare the feature tensor directly before global average pooling
        - add a small BatchNorm scaling term

        Interpretation:
        Higher Zen-Score usually means richer input sensitivity / expressivity at initialization.
        """
        was_training = model.training
        features = []
        hooks = []

        def feature_hook(module, inputs, output):
            if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                features.append(inputs[0].detach())

        try:
            model.eval()

            # Prefer the tensor before global average pooling, because it still contains
            # spatial feature information. Fallback below uses logits if this hook is unavailable.
            if hasattr(model, "global_pool"):
                hooks.append(model.global_pool.register_forward_hook(feature_hook))

            with torch.no_grad():
                noise = torch.randn_like(x)
                x_perturbed = x + epsilon * noise

                output_1 = model(x)
                feature_1 = features[-1] if len(features) > 0 else output_1.detach()

                output_2 = model(x_perturbed)
                feature_2 = features[-1] if len(features) > 0 else output_2.detach()

                feature_diff = torch.mean(torch.abs(feature_1 - feature_2))

                # BatchNorm scaling term, common in Zen-style implementations.
                # For freshly initialized BatchNorm weights this is usually close to zero,
                # but it keeps the score compatible with architectures containing BN.
                bn_term = 0.0
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d) and module.weight is not None:
                        bn_term += torch.log(torch.mean(torch.abs(module.weight.detach())) + 1e-12).item()

                score = torch.log(feature_diff + 1e-12).item() + bn_term

            if not math.isfinite(score):
                return float("-inf")

            return float(score)

        except RuntimeError:
            return float("-inf")

        finally:
            for hook in hooks:
                hook.remove()
            model.train(was_training)

    def fisher_score(self, model, x, target):
        """
        Fisher / empirical Fisher proxy:
        Approximates the diagonal Fisher information at initialization with squared
        gradients from one supervised mini-batch.

        Score:
            sum(gradient ** 2)

        Interpretation:
        Higher Fisher means the model output/loss is more sensitive to its parameters.
        Very small values can indicate weak learning signal or dead paths.

        Note:
        This uses labels from the zero-shot batch, so it is data-dependent.
        """
        was_training = model.training

        try:
            model.eval()
            model.zero_grad(set_to_none=True)

            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()

            score = 0.0
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    score += torch.sum(parameter.grad.detach() ** 2).item()

            if not math.isfinite(score):
                return float("-inf")

            return float(score)

        except RuntimeError:
            return float("-inf")

        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

    def jacobian_score(self, model, x, target=None):
        """
        Jacobian-based proxy:
        Measures how diverse the input gradients are across samples.

        Practical implementation:
        - make the input require gradients
        - compute gradients of selected logits with respect to the input
        - flatten per-sample Jacobians
        - compute logdet of the Jacobian Gram matrix

        Score:
            logdet(J J^T + eps * I)

        Interpretation:
        Higher score means different samples induce more diverse input-output
        sensitivities at initialization. This can be interpreted as a crude
        expressivity/separability proxy.

        If target is given, the logit of the true class is used.
        Otherwise, the currently predicted logit is used.
        """
        was_training = model.training

        try:
            model.eval()
            model.zero_grad(set_to_none=True)

            x_for_grad = x.detach().clone().requires_grad_(True)
            output = model(x_for_grad)

            if target is not None:
                selected_logits = output.gather(1, target.view(-1, 1)).sum()
            else:
                selected_logits = output.max(dim=1).values.sum()

            input_grad = torch.autograd.grad(
                selected_logits,
                x_for_grad,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]

            jacobian = input_grad.flatten(start_dim=1)

            # Normalize each sample's gradient vector. This makes the score less
            # dominated by pure gradient magnitude and more focused on diversity.
            jacobian = jacobian / (torch.norm(jacobian, dim=1, keepdim=True) + 1e-12)

            gram_matrix = jacobian @ jacobian.t()
            batch_size = gram_matrix.shape[0]
            gram_matrix = gram_matrix + 1e-6 * torch.eye(batch_size, device=x.device)

            sign, logdet = torch.linalg.slogdet(gram_matrix)

            if sign <= 0 or not torch.isfinite(logdet):
                return float("-inf")

            return float(logdet.item())

        except RuntimeError:
            return float("-inf")

        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

    def zero_shot_proxy_score(
            self,
            arch,
            model=None,
            x=None,
            target=None,
            channels=16,
            num_cells=NUM_CELLS,
            device=DEVICE,
            enabled_proxies=("naswot", "synflow"),
    ):
        """
        Compute selected zero-shot proxy scores for one architecture.

        enabled_proxies controls which proxies are computed, for example:
            ("naswot", "synflow")
            ("naswot", "synflow", "gradnorm", "snip", "zen", "fisher", "jacobian")

        Returns a dict because the individual scores are normalized later
        across all candidates inside zero_shot_filter().
        """
        owns_model = model is None
        enabled_proxies = tuple(enabled_proxies)

        try:
            if model is None:
                model = self._build_model_from_arch(
                    arch,
                    channels=channels,
                    num_cells=NUM_CELLS,
                    device=device
                )

            if x is None or target is None:
                x, target = self._get_zero_shot_batch(batch_size=16, device=device)

            proxy_scores = {}

            if "naswot" in enabled_proxies:
                proxy_scores["naswot"] = self.naswot_score(model, x)

            if "synflow" in enabled_proxies:
                proxy_scores["synflow"] = self.synflow_score(model, device=device)

            if "gradnorm" in enabled_proxies:
                proxy_scores["gradnorm"] = self.gradnorm_score(model, x, target)

            if "snip" in enabled_proxies:
                proxy_scores["snip"] = self.snip_score(model, x, target)

            if "zen" in enabled_proxies:
                proxy_scores["zen"] = self.zen_score(model, x)

            if "fisher" in enabled_proxies:
                proxy_scores["fisher"] = self.fisher_score(model, x, target)

            if "jacobian" in enabled_proxies:
                proxy_scores["jacobian"] = self.jacobian_score(model, x, target)

            return proxy_scores

        finally:
            if owns_model:
                del model
                #if torch.cuda.is_available():
                    #torch.cuda.empty_cache()

    def _minmax_normalize(self, values):
        """
        Robust min-max normalization.
        Invalid values get 0.0 so broken candidates cannot win.
        """
        finite_values = [v for v in values if math.isfinite(v)]

        if len(finite_values) == 0:
            return [0.0 for _ in values]

        min_value = min(finite_values)
        max_value = max(finite_values)

        if abs(max_value - min_value) < 1e-12:
            return [1.0 if math.isfinite(v) else 0.0 for v in values]

        return [
            (v - min_value) / (max_value - min_value) if math.isfinite(v) else 0.0
            for v in values
        ]

    def zero_shot_filter(
            self,
            architectures,
            top_k,
            channels=16,
            num_cells=NUM_CELLS,
            device=DEVICE,
            enabled_proxies=("naswot", "synflow",),
            proxy_weights=None,
            efficiency_weights=None,
            synflow_drop_fraction=0.10,
            forward_batch_size=16,
    ):
        """
        Central zero-shot filtering pipeline.

        Phase 1:
            Hard Filtering

        Phase 2:
            Selected zero-shot proxy scoring

        Default behavior keeps your previous setup:
            enabled_proxies=("naswot", "synflow")

        Example combinations:
            enabled_proxies=("naswot", "synflow", "gradnorm", "snip", "zen")
            enabled_proxies=("naswot", "fisher", "jacobian")

        proxy_weights:
            Dict with one weight per proxy. Positive weight means "higher is better".

        efficiency_weights:
            Penalties for model cost, e.g. {"flops": 0.10, "n_params": 0.10}
        """
        if architectures is None or len(architectures) == 0:
            return []

        enabled_proxies = tuple(enabled_proxies)

        if proxy_weights is None:
            proxy_weights = {
                "naswot": 1.00,
                "synflow": 0.30,
                "gradnorm": 0.20,
                "snip": 0.20,
                "zen": 0.20,
                "fisher": 0.15,
                "jacobian": 0.15,
            }

        if efficiency_weights is None:
            efficiency_weights = {
                "flops": 0.10,
                "n_params": 0.10,
            }

        top_k = max(1, min(top_k, len(architectures)))
        x, target = self._get_zero_shot_batch(batch_size=forward_batch_size, device=device)

        scored_candidates = []
        rejected_reasons = {}

        for idx, arch in enumerate(architectures):
            model = None

            try:
                model = self._build_model_from_arch(
                    arch,
                    channels=channels,
                    num_cells=NUM_CELLS,
                    device=device
                )

                is_valid, hard_info = self.hard_filter_architecture(
                    arch,
                    model=model,
                    x=x,
                    channels=channels,
                    num_cells=NUM_CELLS,
                    device=device
                )

                if not is_valid:
                    reason = hard_info.get("reason", "unknown")
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                    continue

                proxy_info = self.zero_shot_proxy_score(
                    arch,
                    model=model,
                    x=x,
                    target=target,
                    channels=channels,
                    num_cells=NUM_CELLS,
                    device=device,
                    enabled_proxies=enabled_proxies,
                )

                has_invalid_proxy = False
                for proxy_name in enabled_proxies:
                    proxy_value = proxy_info.get(proxy_name, float("-inf"))
                    if not math.isfinite(proxy_value):
                        reason = f"invalid_{proxy_name}"
                        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                        has_invalid_proxy = True
                        break

                if has_invalid_proxy:
                    continue

                candidate = {
                    "arch": arch,
                    "n_params": hard_info["n_params"],
                    "flops": hard_info["flops"],
                    "hard_info": hard_info,
                }
                candidate.update(proxy_info)
                scored_candidates.append(candidate)

            finally:
                del model
                #if torch.cuda.is_available():
                    # torch.cuda.empty_cache()

        if len(scored_candidates) == 0:
            print("Zero-Shot Filter: No valid architectures found. Falling back to unfiltered candidates.")
            self.last_zero_shot_results = []
            return architectures[:top_k]

        # Optional: remove only the worst SynFlow tail.
        # This is only applied when SynFlow is part of the enabled proxy set.
        if "synflow" in enabled_proxies and len(scored_candidates) > top_k and synflow_drop_fraction > 0:
            before_synflow_filter = len(scored_candidates)

            synflow_values = sorted([candidate["synflow"] for candidate in scored_candidates])
            threshold_index = int(len(synflow_values) * synflow_drop_fraction)
            threshold_index = min(threshold_index, len(synflow_values) - 1)
            synflow_threshold = synflow_values[threshold_index]

            filtered_by_synflow = [
                candidate for candidate in scored_candidates
                if candidate["synflow"] >= synflow_threshold
            ]

            if len(filtered_by_synflow) >= top_k:
                scored_candidates = filtered_by_synflow

                removed_by_synflow = before_synflow_filter - len(scored_candidates)
                rejected_reasons["low_synflow_tail"] = (
                        rejected_reasons.get("low_synflow_tail", 0) + removed_by_synflow
                )

        # Normalize all selected proxy values dynamically.
        normalized_proxy_values = {}
        for proxy_name in enabled_proxies:
            normalized_proxy_values[proxy_name] = self._minmax_normalize(
                [candidate[proxy_name] for candidate in scored_candidates]
            )

        params_norm = self._minmax_normalize([candidate["n_params"] for candidate in scored_candidates])
        flops_norm = self._minmax_normalize([candidate["flops"] for candidate in scored_candidates])

        for candidate_idx, candidate in enumerate(scored_candidates):
            final_score = 0.0

            for proxy_name in enabled_proxies:
                final_score += (
                        proxy_weights.get(proxy_name, 0.0)
                        * normalized_proxy_values[proxy_name][candidate_idx]
                )

            final_score -= efficiency_weights.get("flops", 0.0) * flops_norm[candidate_idx]
            final_score -= efficiency_weights.get("n_params", 0.0) * params_norm[candidate_idx]

            candidate["final_proxy_score"] = final_score

        scored_candidates.sort(key=lambda c: c["final_proxy_score"], reverse=True)

        self.last_zero_shot_results = scored_candidates

        total_rejected = sum(rejected_reasons.values())
        print(
            f"Zero-Shot Filter: {len(scored_candidates)}/{len(architectures)} candidates kept. "
            f"{total_rejected} rejected/removed. Training top {top_k}."
        )

        if len(rejected_reasons) > 0:
            print(f"Rejected candidates by reason: {rejected_reasons}")

        print(f"Enabled proxies: {enabled_proxies}")
        print(f"Proxy weights: {proxy_weights}")
        print(f"Efficiency weights: {efficiency_weights}")

        print("\nTop 5 zero-shot candidates:")
        for rank, candidate in enumerate(scored_candidates[:5], start=1):
            proxy_parts = [
                f"{proxy_name}={candidate[proxy_name]:.4e}"
                for proxy_name in enabled_proxies
            ]

            print(
                f"  #{rank}: "
                f"final={candidate['final_proxy_score']:.4f} | "
                + " | ".join(proxy_parts)
                + f" | Params={candidate['n_params']} | FLOPs={candidate['flops']}"
            )

        return [candidate["arch"] for candidate in scored_candidates[:top_k]]

    # ------------------------------------------------------------------------------------------------------------------
    # TRAINING AND EVALUATION HELPERS
    # ------------------------------------------------------------------------------------------------------------------

    def train_one_epoch(self, model, train_loader, optimizer, device):
        """Train the model for one epoch and return the mean batch loss."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, model, data_loader, device):
        """Return classification accuracy on `data_loader`."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        return correct / max(total, 1)

    def train_and_evaluate(self, arch, train_loader, val_loader, num_epochs=5,
                           lr=0.01, channels=16, num_cells=NUM_CELLS, device=DEVICE
                           ):
        in_channels = self.metadata['input_shape'][1]
        num_classes = self.metadata['num_classes']
        model = NASNetwork(arch, NUM_CELLS, channels=channels,
                           num_classes=num_classes, in_channels=in_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_accs = []
        best_val_acc = 0.0
        best_model_state = None

        for epoch in tqdm(range(num_epochs), desc="Training", leave=False):
            loss = self.train_one_epoch(model, train_loader, optimizer, device)
            train_losses.append(loss)

            val_acc = self.evaluate(model, val_loader, device)
            val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return best_val_acc, model, train_losses, val_accs

    # ------------------------------------------------------------------------------------------------------------------
    # CURRENT ACTIVE SEARCH STRATEGY: RANDOM SEARCH
    # ------------------------------------------------------------------------------------------------------------------



    # --------------------------------------------------
    # HYPERBAND / SUCCESSIVE HALVING - PREPARED, NOT ACTIVE
    # --------------------------------------------------
    def hyperband_search(self,rng, min_budget_per_model=1, max_budget_per_model=27, eta=3, search_time_limit=None):
        print("\n" + "=" * 60)
        print("Starting Hyperband Search...")
        print("=" * 60)
        """The hyperband algorithm

        Parameters
        ----------
        min_budget_per_model : int
            The minimum budget per model

        max_budget_per_model : int
            The maximum budget per model

        eta : float
            The eta parameter. Determines the reduction factor of models.

        Returns
        -------
        TODO
        """
        start_time = time.time()

        x = max_budget_per_model / min_budget_per_model
        s_max = int(math.floor(math.log(x) / math.log(eta)))

        global_results = []
        global_best_arch = None
        global_best_val_acc = -1.0
        global_best_model_state = None
        best_acc_history = []

        iterations = reversed(range(s_max + 1))

        # tqdm gives us a nice progress bar
        for s in tqdm(iterations, desc="Hyperband iter"):

            n_models = int(math.ceil((s_max + 1) / (s + 1) * eta ** s))

            min_budget_per_model_iter = max(1,int(math.ceil(max_budget_per_model / eta ** s)))

            best_state, best_acc, best_arch, bracket_results = self.successive_halving_round(
                rng,
                n_models=n_models,
                min_budget_per_model=min_budget_per_model_iter,
                max_budget_per_model=max_budget_per_model,
                eta=eta,
                search_time_limit=search_time_limit,
                start_time=start_time
            )

            global_results.extend(bracket_results)

            if best_acc > global_best_val_acc:
                global_best_val_acc = best_acc
                global_best_arch = best_arch
                global_best_model_state = best_state

            # best_acc_history.append(global_best_val_acc)

            # If the inner loop timed out (or we naturally crossed the limit during bookkeeping),
            # stop Hyperband immediately before it tries to spin up another bracket.
            if search_time_limit and (time.time() - start_time) >= search_time_limit:
                print("\n⚠️ Hyperband iteration halting immediately to respect search time limit.")
                break

        best_model = None
        if global_best_arch is not None and global_best_model_state is not None:
            in_channels = self.metadata['input_shape'][1]
            num_classes = self.metadata['num_classes']
            best_model = NASNetwork(global_best_arch, NUM_CELLS, channels=16,
                                    num_classes=num_classes, in_channels=in_channels).to(DEVICE)
            best_model.load_state_dict(global_best_model_state)

        best_acc_history = []
        running_best = -1.0
        for r in global_results:
            if r[1] > running_best:
                running_best = r[1]
            best_acc_history.append(running_best)

        self.save_plot(best_model, len(global_results), best_acc_history, global_best_val_acc, global_results,
                       global_best_arch)

        return best_model, global_results, global_best_arch, global_best_val_acc

    def successive_halving_round(self,rng, n_models: int, min_budget_per_model: int, max_budget_per_model: int, eta: float,
                                 search_time_limit: float = None, start_time: float = None):
        """The successive_halving routine as called by hyperband

        Parameters
        ----------
        problem : Problem
            A problem instance to evaluate on

        n_models : int
            How many models to use

        min_budget_per_model : int
            The minimum budget per model

        max_budget_per_model : int
            The maximum budget per model

        eta : float
            The eta parameter. Determines the reduction factor of models.

        random_seed : int | None = None
            The random seed to use

        Returns
        -------
        dict[int, dict]
            A dictionary mapping from the model id as a integer to the config of that model
        """

        in_channels = self.metadata['input_shape'][1]
        num_classes = self.metadata['num_classes']

        # Generate more candidates than we can actually train.
        # Zero-shot filtering chooses the most promising subset.
        # TODO: KI Werte, nochmal checken
        n_candidates = 1000
        top_k_to_train = n_models

        candidate_architectures = [
            self.random_architecture(rng)
            for _ in range(n_candidates)
        ]

        print("time remaining before zero shot filter: ~{}".format(show_time(self.clock.check())))
        architectures_to_train = self.zero_shot_filter(
            candidate_architectures,
            top_k=top_k_to_train,
            channels=16,
            num_cells=NUM_CELLS,
            device=DEVICE,
            forward_batch_size=16
        )
        print("time remaining after zero shot filter: ~{}".format(show_time(self.clock.check())))

        # Safety fallback: at least one architecture should be trained.
        if len(architectures_to_train) == 0:
            architectures_to_train = [self.random_architecture(rng)]

        active_models = {}
        for i in range(n_models):
            active_models[i] = {
                "arch": architectures_to_train[i],
                "model_state": None,
                "optim_state": None,
                "epochs_trained": 0,
                "val_acc": 0.0
            }

        budget = min_budget_per_model
        iteration = 1

        bracket_results = []
        bracket_best_acc = -1.0
        bracket_best_arch = None
        bracket_best_state = None

        while budget <= max_budget_per_model:
            print(
                f"--- SH Iteration {iteration} | Target Budget: {budget} Epochs | Active Models: {len(active_models)}")

            for id_, state in tqdm(active_models.items(), desc="Evaluating", leave=False):
                # Always check the clock
                if self.clock.check() < 60:
                    print("Time is running out! Halting SH bracket early.")
                    break

                if start_time and search_time_limit:
                    elapsed_search = time.time() - start_time
                    if elapsed_search > search_time_limit:
                        print("⚠️ Search time limit 80% reached! Halting SH bracket early to save time for training.")
                        return bracket_best_state, bracket_best_acc, bracket_best_arch, bracket_results

                arch = state["arch"]
                model = NASNetwork(arch, NUM_CELLS, channels=16,
                                   num_classes=num_classes, in_channels=in_channels).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                # Warm-start from previous iteration in this bracket
                if state["model_state"] is not None:
                    model.load_state_dict(state["model_state"])
                    optimizer.load_state_dict(state["optim_state"])

                epochs_to_train = int(budget - state["epochs_trained"])

                # Train the delta epochs
                for _ in range(epochs_to_train):
                    self.train_one_epoch(model, self.train_loader, optimizer, DEVICE)

                state["epochs_trained"] = budget
                val_acc = self.evaluate(model, self.valid_loader, DEVICE)
                state["val_acc"] = val_acc

                # Save state back to RAM
                state["model_state"] = copy.deepcopy(model.state_dict())
                state["optim_state"] = copy.deepcopy(optimizer.state_dict())

                n_param = count_parameters(model)
                bracket_results.append((arch, val_acc, n_param))

                # Track bracket best
                if val_acc > bracket_best_acc:
                    bracket_best_acc = val_acc
                    bracket_best_arch = arch
                    bracket_best_state = copy.deepcopy(model.state_dict())

                # Free VRAM
                del model
                del optimizer
                torch.cuda.empty_cache()

            # Compute configurations to keep
            num_configs_to_proceed = max(1, int(len(active_models) / eta))
            sorted_models = sorted(active_models.items(), key=lambda item: item[1]["val_acc"], reverse=True)
            active_models = dict(sorted_models[:num_configs_to_proceed])

            budget = int(budget * eta)
            iteration += 1

        return bracket_best_state, bracket_best_acc, bracket_best_arch, bracket_results

    # ------------------------------------------------------------------------------------------------------------------
    # RESULT SAVING
    # ------------------------------------------------------------------------------------------------------------------

    def save_plot(self, best_model, n_architectures, best_acc_history, best_val_acc, results, best_arch):
        # ----------------------------------------------------------
        # Plots speichern
        # ----------------------------------------------------------
        os.makedirs("figures", exist_ok=True)

        if len(results) == 0:
            print("No results to plot.")
            return

        # --- Plot 1: Verlauf der Genauigkeit über die Suche ---
        iterations = list(range(1, len(results) + 1))
        raw_accs = [r[1] for r in results]
        codename = self.metadata["codename"]
        plt.figure(figsize=(8, 5))
        plt.plot(iterations, raw_accs, marker='o', linestyle='-', color='gray',
                 alpha=0.6, label='Val Acc pro Architektur')
        plt.plot(iterations, best_acc_history, marker='o', linestyle='-', color='red',
                 linewidth=2, label='Bester Wert bisher')
        plt.xlabel('Architektur (Iteration)')
        plt.ylabel('Validation Accuracy')
        plt.title('Verlauf der Genauigkeit während der Random Search')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"figures/{codename}_accuracy_progress{codename}.png")
        plt.close()

        # --- Plot 2: Accuracy vs. Parameteranzahl ---
        params = [r[2] for r in results]
        accs = [r[1] for r in results]

        plt.figure(figsize=(8, 5))
        plt.scatter(params, accs, c='blue', label='Gefundene Architekturen')

        if best_model is not None:
            best_params = count_parameters(best_model)
            plt.scatter([best_params], [best_val_acc], c='red', s=100,
                        label='Beste Architektur', zorder=5)
        plt.xlabel('Anzahl Parameter')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy vs. Parameteranzahl')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"figures/{codename}_random_search_results.png")
        plt.close()

        # ----------------------------------------------------------
        # Rohdaten speichern
        # ----------------------------------------------------------
        os.makedirs("results", exist_ok=True)

        # --- CSV: eine Zeile pro Architektur (ohne volle Architektur-Details) ---
        with open(f"results/{codename}_random_search_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "val_acc", "best_val_acc_so_far", "n_params"])
            for idx, (arch, val_acc, n_param) in enumerate(results, start=1):
                writer.writerow([idx, float(val_acc), float(best_acc_history[idx - 1]), int(n_param)])

        # --- JSON: volle Details inkl. Architektur selbst ---
        json_results = []
        for idx, (arch, val_acc, n_param) in enumerate(results, start=1):
            # Tupel-Keys (i, j) -> String "i_j", damit JSON-kompatibel
            arch_serializable = {f"{i}_{j}": op for (i, j), op in arch.items()}
            json_results.append({
                "iteration": idx,
                "architecture": arch_serializable,
                "val_acc": float(val_acc),
                "n_params": int(n_param),
            })

        best_arch_serializable = None
        if best_arch is not None:
            best_arch_serializable = {f"{i}_{j}": op for (i, j), op in best_arch.items()}

        with open(f"results/{codename}_random_search_results.json", "w") as f:
            json.dump({
                "results": json_results,
                "best_architecture": best_arch_serializable,
                "best_val_acc": float(best_val_acc),
            }, f, indent=2)


# ======================================================================================================================
# SEARCH SPACE MODEL
# ======================================================================================================================

class NASNetwork(nn.Module):
    def __init__(self, arch, num_cells=NUM_CELLS, channels=16, num_classes=10, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.cells = nn.ModuleList()
        for c in range(num_cells):
            current_channels = 8 if c == 0 else channels
            self.cells.append(NASCell(arch, current_channels))

            # If we are not on the last cell, add a transition block
            if c < num_cells - 1:
                self.cells.append(nn.Sequential(
                    # FIX: Use current_channels instead of hardcoding 8
                    nn.Conv2d(current_channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class NASCell(nn.Module):
    """DAG cell: each subsequent node sums all incoming edge outputs."""

    def __init__(self, arch, channels):
        super().__init__()
        self.arch = arch
        self.channels = channels
        self.edge_ops = nn.ModuleDict()
        for (i, j), op_name in arch.items():
            key = f"{i}_{j}"
            self.edge_ops[key] = make_operation(op_name, channels)

    def forward(self, x):
        node_outputs = {0: x}
        for j in range(1, NUM_NODES):
            inputs = []
            for i in range(j):
                key = f"{i}_{j}"
                edge_out = self.edge_ops[key](node_outputs[i])
                inputs.append(edge_out)
            node_outputs[j] = sum(inputs)
        return node_outputs[NUM_NODES - 1]


# ======================================================================================================================
# SEARCH SPACE OPERATIONS
# ======================================================================================================================

def make_operation(op_name, channels):
    if op_name == "conv3x3":
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
    elif op_name == "conv5x5":
        return nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
    elif op_name == "maxpool3x3":
        return nn.MaxPool2d(3, stride=1, padding=1)
    elif op_name == "avgpool3x3":
        return nn.AvgPool2d(3, stride=1, padding=1)
    elif op_name == "skip":
        return nn.Identity()
    elif op_name == "none":
        return ZeroOp()
    #added
    elif op_name == "sep_conv3x3":
        return sep_conv(channels, kernel_size=3, padding=1)

    elif op_name == "sep_conv5x5":
        return sep_conv(channels, kernel_size=5, padding=2)

    elif op_name == "dil_conv3x3":
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    elif op_name == "dil_sep_conv3x3":
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    elif op_name == "conv1x1":
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    elif op_name == "factorized_conv5x5":
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 5),
                padding=(0, 2),
                bias=False,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(5, 1),
                padding=(2, 0),
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    elif op_name == "avgpool3x3_conv1x1":
        return nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    else:
        raise ValueError(f"Unknown operation: {op_name}")


class ZeroOp(nn.Module):
    """Operation that outputs zeros (removes the edge)."""

    def forward(self, x):
        return torch.zeros_like(x)

def sep_conv(channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        ),
        nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        ),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )

# ======================================================================================================================
# UTILS
# ======================================================================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# === TIME COUNTERs ====================================================================================================
def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)
