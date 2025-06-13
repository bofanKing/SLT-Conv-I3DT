import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from signjoey.i3d_embedding import I3DEmbedding
from signjoey.i3d_video_dataset import I3DVideoDataset


class I3DFeatureExtractor:
    def __init__(self, base_root, filelist_root, output_base, output_dim=1024, device=None):
        self.base_root = base_root
        self.filelist_root = filelist_root
        self.output_base = output_base
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = I3DEmbedding(output_dim=output_dim, pretrained=True).to(self.device)
        self.model.eval()

    def collate_fn(self, batch):
        clips, names = zip(*batch)
        return list(clips), list(names) # 帧虚列 和 名称列表

    def extract_for_split(self, split):
        print(f"\n🧪 开始处理 {split.upper()} 数据...")
        #input("输入开始数据处理")

        frame_root = os.path.join(self.base_root, split)
        file_list_path = os.path.join(self.filelist_root, f"phoenix14t_i3d_filelist.{split}.sample2.txt")
        output_dir = os.path.join(self.output_base, f"i3d_features_{split}_sample2")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(frame_root):
            print(f"❌ 帧路径不存在: {frame_root}")
            return
        if not os.path.exists(file_list_path):
            print(f"❌ 文件列表不存在: {file_list_path}")
            return

        dataset = I3DVideoDataset(frame_root=frame_root, file_list=file_list_path, num_frames=None)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                            collate_fn=self.collate_fn, drop_last=False)

        for clips, names in tqdm(loader, desc=f"Extracting I3D features for {split}"):
            for clip, name in zip(clips, names):
                clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # [1, 3, T, H, W] 加入B=1，统一格式
                print(f"🟢 输入 shape: {clip.shape}")
                with torch.no_grad(): # 只在乎前向提取 不在乎推理过程
                    feat = self.model(clip)  # [T', 1024]
                print(f"🟢 提取后 shape: {feat.shape}")
                torch.save([f for f in feat], os.path.join(output_dir, f"{name.replace('/', '_')}.pt"))

        print(f"✅ 特征提取完成，保存至 {output_dir}")

    def run(self, splits):
        for split in splits:
            self.extract_for_split(split)


# ✅ 主程序入口
if __name__ == "__main__":
    extractor = I3DFeatureExtractor(
        base_root="../PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px",
        filelist_root="../data/PHOENIX2014T",
        output_base="../"
    )
    extractor.run(["dev"])  # 你可以改为 ["train", "dev", "test"]  # 目前都替换完了 记得 train你训练了11h
