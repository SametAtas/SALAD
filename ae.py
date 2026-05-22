import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, d_k=None, n_groups=32):
        super().__init__()
        if d_k is None:
            d_k = channels // num_heads
        self.norm = nn.GroupNorm(n_groups, channels)
        self.projection = nn.Linear(channels, num_heads * d_k * 3)
        self.output = nn.Linear(num_heads * d_k, channels)
        self.scale = d_k**-0.5
        self.num_heads = num_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.num_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        res = res.reshape(batch_size, -1, self.num_heads * self.d_k)
        res = self.output(res)

        res += x

        res = res.permute(0, 2, 1).reshape(batch_size, channels, height, width)

        return res


class AutoEncoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        parameters = {
                "parameters": {
                "n_channels": 128,
                "ch_mults": [1,2,2,2],
                "is_attn": [False,False,False,False],
                "n_blocks": 1,
                "image_channels": parameters.get("image_channels", 6),
                "out_channels": parameters.get("out_channels", 6),
                "append_positional_embedding_start": False,
                "add_positional_embedding_start": False
            },
        }
        self.enc = Encoder(parameters["parameters"])
        self.dec = Decoder(parameters["parameters"])
        self.bottleneck_attn = SelfAttentionBlock(channels=1024, num_heads=8)


    def forward(self, x):
        x = self.enc(x)
        x = self.bottleneck_attn(x)
        x = self.dec(x)
        return x


class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        ch_mults = parameters["ch_mults"]
        n_channels = parameters["n_channels"]
        image_channels = parameters["image_channels"]
        n_blocks = parameters["n_blocks"]
        draem = parameters.get("draem",False)
        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv2d(
            image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        down = []
        in_channels = n_channels
        out_channels = in_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels,draem=draem))
                in_channels = out_channels
            if i != n_resolutions - 1:
                down.append(Downsample(out_channels,draem=draem))
        #down.append(DownBlock(out_channels, out_channels))
        self.down = nn.ModuleList(down)

    def forward(self, x):
        x = self.image_proj(x)
        for m in self.down:
            x = m(x)
        return x


class Decoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        
        ch_mults = parameters["ch_mults"]
        n_channels = parameters["n_channels"]
        image_channels = parameters["out_channels"]
        n_blocks = parameters["n_blocks"]
        draem = parameters.get("draem",False)
        
        n_resolutions = len(ch_mults)

        up = []
        in_channels = n_channels
        for i in range(n_resolutions):
            in_channels = in_channels * ch_mults[i]
        out_channels = in_channels

        for i in reversed(range(n_resolutions)):
            for j in range(n_blocks):
                if j == n_blocks - 1:
                    out_channels = out_channels // ch_mults[i]
                
                up.append(UpBlock(in_channels, out_channels,draem=draem))
                in_channels = out_channels
            if i != 0:
                up.append(Upsample(out_channels))
        #up.append(UpBlock(out_channels, out_channels))
        self.up = nn.ModuleList(up)

        self.final = nn.Conv2d(
            n_channels, image_channels, kernel_size=(3, 3), padding=(1, 1)
        )

    def forward(self, x):
        for m in self.up:
            x = m(x)
        x = self.final(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, draem=False):
        super().__init__()
        # self.block = ResidualBlock(in_channels, out_channels)
        self.block = ConvBlock(in_channels, out_channels,draem=draem)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, draem=False) -> None:
        super().__init__()
        # self.block = ResidualBlock(in_channels, out_channels)

        self.block = ConvBlock(in_channels, out_channels, draem=draem)

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    def __init__(self, n_channels,draem=False):
        super().__init__()
        # self.conv = nn.Conv2d(
        #     n_channels, n_channels, (3, 3), (2, 2), (1, 1)
        # )
        self.conv = nn.Sequential(nn.MaxPool2d(2))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, n_channels, draem=False):
        super().__init__()
        # self.conv = nn.ConvTranspose2d(
        #     n_channels, n_channels, (4, 4), (2, 2), (1, 1)
        # )
        if draem:
            self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, n_channels),
                nn.SiLU(),
            )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_groups=16,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        # features = None
        return h + self.shortcut(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final_stride=(1,1), draem=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        if draem:
            self.act = nn.ReLU()
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            self.act = nn.SiLU()
            self.norm1 = nn.GroupNorm(8, in_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), stride=final_stride, padding=(1, 1)
        )
        

    def forward(self, x):
        x = self.conv1(self.act(self.norm1(x))) # Use for larger perceptive field - might be useful in the long run
        return self.conv2(self.act(self.norm2(x)))
