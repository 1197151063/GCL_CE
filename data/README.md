# 项目内数据

`yelp2018/`、`amazon-book/`、`gowalla/`、`iFashion/` 各自的 train.txt、test.txt 和 SHA256SUMS 均纳入本项目的 Git，可随 clone、fetch 或 Git bundle 一起传输。运行时无需访问另一个 R-Rec 目录。

数据来自本机原始 R-Rec 的对应目录，逐字节复制，原始文件未修改。每个目录的 SHA256SUMS 保存两份文件的 SHA-256 校验值；不复制 notebook 缓存或本实验不使用的映射文件。

在 Linux 上可校验：

```bash
cd data/yelp2018
sha256sum -c SHA256SUMS
```

程序仍按固定 split seed 从 train.txt 内划分验证集；不要把 test.txt 当作验证集。训练结果与 checkpoint 保存在项目内的 `runs/`，不纳入 Git 和服务器安装包。
