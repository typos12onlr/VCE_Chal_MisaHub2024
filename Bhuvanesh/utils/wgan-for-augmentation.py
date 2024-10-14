# %% [code] {"execution":{"iopub.status.busy":"2024-10-14T16:26:03.128901Z","iopub.execute_input":"2024-10-14T16:26:03.129871Z","iopub.status.idle":"2024-10-14T16:26:06.478705Z","shell.execute_reply.started":"2024-10-14T16:26:03.129815Z","shell.execute_reply":"2024-10-14T16:26:06.477396Z"}}
import torch
import torch.nn as nn

# %% [code] {"execution":{"iopub.status.busy":"2024-10-14T16:26:06.481195Z","iopub.execute_input":"2024-10-14T16:26:06.481868Z","iopub.status.idle":"2024-10-14T16:26:06.494273Z","shell.execute_reply.started":"2024-10-14T16:26:06.481810Z","shell.execute_reply":"2024-10-14T16:26:06.492813Z"}}
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is a random noise vector of size nz
            # Output: (ngf * 16) x 4 x 4 (initial layer is doubled for larger image)
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            # Output: (ngf * 8) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # Output: (ngf * 4) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # Output: (ngf * 2) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Output: (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Output: (nc) x 256 x 256
            nn.ConvTranspose2d(ngf, nc, 4, 4, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-14T16:26:06.496961Z","iopub.execute_input":"2024-10-14T16:26:06.497582Z","iopub.status.idle":"2024-10-14T16:26:06.513646Z","shell.execute_reply.started":"2024-10-14T16:26:06.497499Z","shell.execute_reply":"2024-10-14T16:26:06.511743Z"}}
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 4, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)

# %% [code]
