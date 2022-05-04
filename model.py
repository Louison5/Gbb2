# Model part
import torch
from torch import nn
class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans,
                              kernel_size=3, padding=1,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)  # <5>
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')  # <6>
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)  # <7>
        torch.nn.init.zeros_(self.batch_norm.bias)

        self.conv2 = nn.Conv2d(n_chans, n_chans,
                              kernel_size=3, padding=1,
                              bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=n_chans)  # <5>
        torch.nn.init.kaiming_normal_(self.conv2.weight,
                                      nonlinearity='relu')  # <6>
        torch.nn.init.constant_(self.batch_norm2.weight, 0.5)  # <7>
        torch.nn.init.zeros_(self.batch_norm2.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        out = self.conv2(x)
        out = self.batch_norm2(out)
        return torch.relu(out + x)
class CNNModel(nn.Module):


    def __init__(self,  n_blocks=9):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(144, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.batch_norm=nn.BatchNorm2d(64)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=64)])
        )  # <8>
        self.fc = nn.Linear(9 * 4* 64, 235)

  
    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        action_logits=self.fc(self.resblocks(torch.relu(self.batch_norm(self.conv2(self.conv1(obs))))).view(-1,9 * 4* 64))
        # action_logits = self._tower(obs)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask
    # def forward(self, x):
    #     out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
    #     out = self.resblocks(out)
    #     out = F.max_pool2d(out, 2)
    #     out = out.view(-1, 8 * 8 * self.n_chans1)
    #     out = torch.relu(self.fc1(out))
    #     out = self.fc2(out)
    #     return out
    # def __init__(self):
    #     nn.Module.__init__(self)
    #     self._tower = nn.Sequential(
    #         nn.Conv2d(6, 64, 3, 1, 1, bias = False),
    #         nn.ReLU(True),
    #         nn.Conv2d(64, 64, 3, 1, 1, bias = False),
    #         nn.ReLU(True),
    #         nn.Conv2d(64, 64, 3, 1, 1, bias = False),
    #         nn.ReLU(True),
    #         nn.Flatten(),
    #         nn.Linear(64 * 4 * 9, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 235)
    #     )
        
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight)
model=CNNModel()
torch.save(model.state_dict(),"data/your_model_name.pkl")