# Add Lecture Site

这是一个无需构建的静态讲义网站。发布目录中的 `index.html` 会读取同目录下的四份 Markdown 讲义。

## Cloudflare Pages

1. 将整个 `add_lecture_site` 目录推送到 GitHub 仓库。
2. 在 Cloudflare Pages 新建项目并连接该仓库。
3. 设置 `Framework preset` 为 `None`，`Build command` 留空，`Build output directory` 填写 `add_lecture_site`。
4. 部署完成后，将得到一个可公开访问的 `*.pages.dev` 地址。

## GitHub Pages

将 `add_lecture_site` 作为仓库根目录，或将它移动为仓库中的 `docs` 目录。随后在仓库 `Settings > Pages` 中选择对应分支和目录作为发布源。
