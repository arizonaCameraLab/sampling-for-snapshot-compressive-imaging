import torch
import torch.nn as nn

class PATvidx4(nn.Module):
    """    
    Suppose we plan to reconstruct 3 channel high resolution video frame n, which is (cout, h, w).
    (Note that cout=3, leave it as a parameter in case we need future adjustment.)
    The input contains 4 images. 
    (Note that lr_cin=1, hr_cin=1, leave these as parameters in case we need future adjustment.)
        (lr_cin, h//4, w//4), frame n  , second channel (middle frame, middle channel)
        (lr_cin, h//4, w//4), frame n-1, first channel
        (lr_cin, h//4, w//4), frame n+1, third channel
        (hr_cin, h,    w   ), frame n+d, grayscale. d=-4, -3, ..., 3, 4
    The channel order of the output frame depends on the input frames, and 
    the alpha frame is always the first frame, providing geometry guidance
    """
    def __init__(self, *, lr_cin=1, hr_cin=1, cout=3):
        super().__init__()
        
        ### feature extraction blocks
        self.lr_feature = nn.Sequential(
            nn.Conv2d(lr_cin, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResB(64),
            ResASPPB(64),
            ResB(64),
            ResASPPB(64),
            ResB(64),
        )
        self.hr_feature = nn.Sequential(
            nn.Conv2d(hr_cin, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResB(64),
            ResASPPB(64),
            ResB(64),
            nn.Conv2d(64, 96, 3, 2, 1, bias=False),
            ResB(96),
            ResASPPB(96),
            ResB(96),
            nn.Conv2d(96, 128, 3, 2, 1, bias=False),
            ResB(128),
            ResASPPB(128),
            ResB(128),
        )
        
        ### parallax attention module (its not parallax anymore, in this case)
        self.pam = PAMvidx4(64, 128, 128, 32)
        
        ### upscaling
        self.upscale = nn.Sequential(
            ResB(128),
            ResB(128),
            ResB(128),
            ResB(128),
            nn.Conv2d(128, 32*(4**2), 1, 1, 0, bias=False),
            nn.PixelShuffle(4),
            ResB(32),
            ResB(32),
            nn.Conv2d(32, cout, 3, 1, 1, bias=False)
        )
        
    def forward(self, lr_ins, hr_in, is_training, Pos):
        ### feature extraction
        lr_buffers = [self.lr_feature(lr_in) for lr_in in lr_ins]
        hr_buffer = self.hr_feature(hr_in)
        if is_training == 1:
            ### parallax attention
            buffer, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = self.pam(lr_buffers, hr_buffer, is_training, Pos)
            ### upscaling
            out = self.upscale(buffer)
            return out, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
                   (V_left_to_right, V_right_to_left)
        if is_training == 0:
            ### parallax attention
            buffer, M_right_to_left = self.pam(lr_buffers, hr_buffer, is_training, Pos)
            ### upscaling
            out = self.upscale(buffer)
            return out, M_right_to_left
        
        
class PAMvidx4(nn.Module):
    """
    Parallax attention module for vidx4 input
    the first input is considered as the alpha input, giving geometry guidance
    """
    def __init__(self, lr_cin, hr_cin, cout, ctoken):
        """
        lr_cin: low-res input channel amount
        hr_cin: high-res input channel amount
        cout: output channel amount
        ctoken: Q/K channel amount
        """
        super().__init__()
        self.lr_rb = ResB(lr_cin)
        self.hr_rb = ResB(hr_cin)
        self.lr_q = nn.Conv2d(lr_cin, ctoken, 1, 1, 0, bias=True)
        self.lr_k = nn.Conv2d(lr_cin, ctoken, 1, 1, 0, bias=True)
        self.hr_k = nn.Conv2d(hr_cin, ctoken, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.fe_pam = fePAM()
        self.fusion = nn.Conv2d(lr_cin*3+hr_cin, cout, 1, 1, 0, bias=True)
        
    def __call__(self, lr_ins, hr_in, is_training, Poss):
        assert len(lr_ins)==3
        
        # preprocess input features
        b, _, h, w = lr_ins[0].shape
        lr_buffers = [self.lr_rb(lr_in) for lr_in in lr_ins]
        hr_buffer = self.hr_rb(hr_in)
        
        # prepare Q, S(K), R(V)
        Q = self.lr_q(lr_buffers[0])
        Ss, Rs = [], []
        for i in range(1, 3):
            Ss.append(self.lr_k(lr_buffers[i]))
            Rs.append(lr_ins[i])
        Ss.append(self.hr_k(hr_buffer))
        Rs.append(hr_in)
        
        # attention process Q, S(K), R(V), aggregate R(V)
        buffers = []
        M_right_to_lefts = []
        for S, R, Pos in zip(Ss, Rs, Poss):
            buffer, M_right_to_left = self.fe_pam(Q, S, R, Pos)
            buffers.append(buffer)
            M_right_to_lefts.append(M_right_to_left)
        buffers.append(lr_buffers[0])
        
        # stack channels and fuse
        out = self.fusion(torch.cat(tuple(buffers), 1))

        ## output
        if is_training == 1:
            return out, (None, None), (None, None), (None, None)
        if is_training == 0:
            return out, M_right_to_lefts
        
class fePAM(nn.Module):
    """
    Free Epipolar Parallax Attention Module
    """
    def __init__(self):
        super(fePAM, self).__init__()
        self.softmax = nn.Softmax(-1)
    def forward(self, Q, S, R, Pos):
        """
        Q: querying token map, n_batch x C x h x w
        S: key token map, n_batch x C x H x W
        R: value token map, n_batch x C' x H x W
        Pos: perceptual field coordinates. 
            xxs: nparray, n_batch x h x w x k; yys: nparray, n_batch x h x w x k
        """
        n_batch, n_channel, h, w = Q.size()
        _, v_channel, _, _ = R.size() # value channels can be different
        Key = []
        Value = []
        for i in range(n_batch):
            Pos_x, Pos_y = Pos[i][0].flatten().long(), Pos[i][1].flatten().long() #(h*w*k, )
            Key.append(S[i, :, Pos_x, Pos_y]) 
            Value.append(R[i, :, Pos_x, Pos_y])
        Key = torch.stack(Key, dim=0) #n_batch x C x h*w*k
        Value = torch.stack(Value, dim=0) #n_batch x C' x h*w*k
        # this process requires to assigning quite a lot of memories. Quite cubersum and large

        Key = Key.view(n_batch, n_channel, h*w, -1).permute(0, 2, 1, 3) #n_batch x h*w x C x k
        Q = Q.permute(0, 2, 3, 1).view(n_batch, h*w, n_channel).unsqueeze(2) # n_batch x h*w x 1 x C
        score = torch.matmul(Q, Key) #n_batch x h*w x 1 x k
        M_right_to_left = self.softmax(score) #n_batch x h*w x 1 x k
        
        Value = Value.view(n_batch, v_channel, h*w, -1).permute(0, 2, 3, 1) #n_batch x h*w x k x C'
        buffer = torch.matmul(M_right_to_left, Value) #n_batch x h*w x 1 x C'
        buffer = buffer.squeeze().view(n_batch, h, w, v_channel).permute(0, 3, 1, 2) #n_batch x C' x h x w

        return buffer, M_right_to_left
    
class ResB(nn.Module):
    """
    Residual block, a very simple implementation
    Learned from PASSRnet, https://github.com/The-Learning-And-Vision-Atelier-LAVA/PASSRnet
    """
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class ResASPPB(nn.Module):
    """
    Residual Atrous Spatial Pyramid Pooling
    Learned from PASSRnet, https://github.com/The-Learning-And-Vision-Atelier-LAVA/PASSRnet
    """
    def __init__(self, channels):
        super(ResASPPB, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.b_1 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_2 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_3 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))

        buffer_2 = []
        buffer_2.append(self.conv1_2(buffer_1))
        buffer_2.append(self.conv2_2(buffer_1))
        buffer_2.append(self.conv3_2(buffer_1))
        buffer_2 = self.b_2(torch.cat(buffer_2, 1))

        buffer_3 = []
        buffer_3.append(self.conv1_3(buffer_2))
        buffer_3.append(self.conv2_3(buffer_2))
        buffer_3.append(self.conv3_3(buffer_2))
        buffer_3 = self.b_3(torch.cat(buffer_3, 1))

        return x + buffer_1 + buffer_2 + buffer_3