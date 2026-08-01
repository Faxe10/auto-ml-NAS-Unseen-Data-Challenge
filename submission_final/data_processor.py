import torch
import torchvision.transforms as transforms
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = torch.as_tensor(x).float()


        # Format [Batch, Channels, Height, Width] erzwingen
        if len(self.x.shape) == 3:
            self.x = torch.reshape(self.x, (self.x.shape[0], 1, self.x.shape[1], self.x.shape[2]))

        if y is None:
            self.y = None
        else:
            self.y = torch.tensor(y).long()

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        im = self.x[idx]
        if self.transform is not None:
            im = self.transform(im)
        if self.y is None:
            return im
        return im, self.y[idx]


class DataProcessor:
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock

    def process(self):
        # 1. Bildmaße aus den Metadaten auslesen
        # metadata['input_shape'] ist typischerweise (N, C, H, W)
        _, channels, orig_h, orig_w = self.metadata['input_shape']

        # Prüfen, ob Downscaling auf 256x256 nötig ist
        transform_list = []
        target_h, target_w = orig_h, orig_w
        if orig_h == orig_w:
            if orig_h > 256:
                target_h, target_w = 256, 256
                transform_list.append(transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR))

        # 2. Temporärer Tensor für die korrekte Statistik-Berechnung erstellen
        tmp_x = torch.as_tensor(self.train_x).float()
        if len(tmp_x.shape) == 3:
            tmp_x = torch.reshape(tmp_x, (tmp_x.shape[0], 1, tmp_x.shape[1], tmp_x.shape[2]))

        # Falls herunterskaliert werden muss, wenden wir das hier für die Statistik vorab an
        if transform_list:
            resizer = transform_list[0]
            tmp_x = resizer(tmp_x)

        # 3. Mittelwert und Standardabweichung pro Kanal berechnen
        mean = torch.mean(tmp_x, dim=[0, 2, 3]).tolist()
        std = torch.std(tmp_x, dim=[0, 2, 3])
        std = torch.clamp(std, min=1e-6).tolist()
        del tmp_x #dierekt loescht da eventuell viel ram ...
        # 4. Normalisierung zur Pipeline hinzufügen
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        pipeline = transforms.Compose(transform_list)

        # 5. Datasets erstellen
        train_ds = Dataset(self.train_x, self.train_y, transform=pipeline)
        valid_ds = Dataset(self.valid_x, self.valid_y, transform=pipeline)
        test_ds = Dataset(self.test_x, None, transform=pipeline)

        # Dynamische Batchgröße: Bei 256x256 Bildern sind 64 oft zu viel für den VRAM.
        # Wenn herunterskaliert wurde, nutzen wir Batchgröße 32, sonst lassen wir sie bei 64.
        if (target_h >= 256 or target_w >= 256 ):
            max_batch = 128
        elif (target_h >= 128 or target_w >= 128):
            max_batch = 256
        else:
            max_batch = 1240

        n_train = len(train_ds)
        n_valid = len(valid_ds)
        batch_size = min(max_batch, n_train, n_valid) if min(n_train, n_valid) > 0 else 1
        batch_size = max(batch_size, 2)

        # Dataloader aufbauen
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, drop_last=False, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # 6. WICHTIG: Metadaten für das NAS-Modell aktualisieren
        self.metadata['input_shape'] = (self.metadata['input_shape'][0], channels, target_h, target_w)

        return train_loader, valid_loader, test_loader
