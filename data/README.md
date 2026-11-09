# 项目内数据

`yelp2018/train.txt` 和 `yelp2018/test.txt` 已包含在本项目中，可随 Git 或服务器压缩包一起传输。运行时无需访问另一个 R-Rec 目录。

数据来自本机原始 R-Rec 的 Yelp2018，逐字节复制，原始文件未修改。`yelp2018/SHA256SUMS` 保存两份文件的 SHA-256 校验值。

在 Linux 上可校验：

```bash
cd data/yelp2018
sha256sum -c SHA256SUMS
```

程序仍按固定 split seed 从 train.txt 内划分验证集；不要把 test.txt 当作验证集。训练结果与 checkpoint 保存在项目内的 `runs/`，不纳入 Git 和服务器安装包。
