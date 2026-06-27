---
layout: post
title: "Fedora 44 Workstation 配置记录"
date: 2026-06-27
author: "LH"
catalog: true
tags:
  - "Linux"
keywords:
  - "Fedora 44"
  - "GNOME"
  - "Wayland"
  - "Fcitx5"
  - "libinput"
group: vision
---

## 引言

最近给笔记本装了 Fedora 44 Workstation，打算把日常开发环境从 Windows 迁移过来。选 Fedora 的理由很简单：dnf 包管理够现代，GNOME 体验最正宗，而且不像 Arch 那样需要花大量时间从零搭积木。

不过"开箱即用"只是理想状态。实际配置过程中踩了不少坑——输入法、触控板滚动速度、桌面图标，这些在 Windows 下从来不用操心的东西，到了 Linux 桌面每一个都得手动调教。

本文记录这次 Fedora GNOME 桌面初始化过程中遇到的问题、原因和处理方法，方便以后重装或排查时复用。

## 系统环境

```text
系统：Fedora 44
桌面：GNOME 50
会话：Wayland
系统语言：中文 zh_CN.UTF-8
用户目录：保持英文路径（Desktop、Downloads、Documents 等）
```

> 用户目录保持英文是个好习惯。如果系统弹出"是否根据当前语言重命名标准文件夹"，一定要选"保留旧名称"。界面仍是中文，但终端里 `cd ~/Desktop` 不用打中文路径。

## 软件安装

日常开发三件套，没什么特别的：

```bash
# Google Chrome（仓库已预置）
dnf install google-chrome-stable

# VS Code
dnf install 'https://code.visualstudio.com/sha/download?build=stable&os=linux-rpm-x64'

# Obsidian（通过 Flathub）
flatpak install flathub md.obsidian.Obsidian
```

另外装了 Extension Manager 方便后续管理 GNOME 扩展：

```bash
flatpak install flathub com.mattjakeman.ExtensionManager
```

> Flatpak 安装的应用启动命令都是 `flatpak run <appid>`，比如 `flatpak run md.obsidian.Obsidian`。

## 输入法：从 IBus 迁移到 Fcitx5

系统自带的 IBus + libpinyin 中文输入体验不太行，换 Fcitx5 是 Linux 桌面的标准操作了。

### 安装

```bash
dnf install fcitx5 fcitx5-autostart fcitx5-chinese-addons fcitx5-configtool fcitx5-gtk fcitx5-gtk3 fcitx5-gtk4 fcitx5-qt fcitx5-rime
```

用户会话环境变量，写入 `~/.config/environment.d/90-fcitx5.conf`：

```ini
GTK_IM_MODULE=fcitx
QT_IM_MODULE=fcitx
XMODIFIERS=@im=fcitx
```

默认输入法配置，写入 `~/.config/fcitx5/profile`：

```ini
[Groups/0]
Name=默认
Default Layout=us
DefaultIM=pinyin

[Groups/0/Items/0]
Name=keyboard-us
Layout=

[Groups/0/Items/1]
Name=pinyin
Layout=

[GroupOrder]
0=默认
```

### 拼音预编辑光标位置

装完后发现一个问题：输入 `ceshi` 时预编辑光标卡在拼音开头——

```text
|ceshi
```

回车提交后倒是正常（`测试|`），但输入过程中完全不知道光标在哪。

原因是 Fcitx5 拼音配置里有个选项把光标固定在了开头。先停掉 Fcitx5（否则退出时会覆盖配置），然后修改 `~/.config/fcitx5/conf/pinyin.conf`：

```bash
fcitx5-remote -e
```

```ini
PreeditMode="Composing pinyin"
PreeditCursorPositionAtBeginning=False
PinyinInPreedit=True
```

重新启动：

```bash
fcitx5 &
```

### 像 Windows 一样用 Ctrl 切换中英文

在 Windows 下习惯按 Ctrl 切换中英文输入，Fcitx5 可以做到。编辑 `~/.config/fcitx5/config`：

```ini
[Hotkey]
TriggerKeys=Control+space Control_L Control_R

[Hotkey/TriggerKeys]
0=Control+space
1=Control_L
2=Control_R
```

如果发现左右 Ctrl 跟应用内快捷键冲突，可以只保留右 Ctrl：

```ini
TriggerKeys=Control+space Control_R
```

## 触控板

### 指针速度 vs 滚动速度

一开始觉得触控板太快，去设置里把 touchpad speed 调低了，结果指针移动变慢但滚动还是一样快。后来才搞清楚：GNOME 50 + Wayland 的 `touchpad speed` 只影响指针移动，不影响两指滚动速度。系统设置里根本没有独立的滚动速度滑块。

恢复默认指针速度：

```bash
gsettings set org.gnome.desktop.peripherals.touchpad speed 0.0
```

当前触控板设备：

```text
MSFT0001:02 06CB:7F28 Touchpad
```

![fedora-workstation](/img/vision/fedora-setup/cover.jpg)

### libinput-config：解决滚动过快

既然系统不给调，就得自己来。参考了 [touchpad-sensitivity-tweak](https://github.com/shivasai573/touchpad-sensitivity-tweak) 项目的思路——安装 `libinput-config`，通过 `/etc/libinput.conf` 设置 `scroll-factor`。

但没有直接跑它的一键脚本，因为脚本会写入 `natural-scroll=disabled`，覆盖掉 GNOME 当前开启的自然滚动。实际采用的是[上游源码](https://gitlab.com/warningnonpotablewater/libinput-config.git)手动编译。

> **关于 `pkexec`**：在图形桌面下需要管理员权限时，用 `pkexec` 会弹出系统授权窗口。写入系统文件时搭配 `sh -c`，需要保留代理环境变量时用 `env` 显式传入。

安装编译依赖（显式使用本地代理）：

```bash
pkexec env http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 \
  HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 \
  dnf install -y --setopt=max_parallel_downloads=3 \
  meson ninja-build libinput-devel gcc gcc-c++ make
```

构建并安装：

```bash
git clone https://gitlab.com/warningnonpotablewater/libinput-config.git /tmp/libinput-config-upstream
cd /tmp/libinput-config-upstream
meson setup build
ninja -C build
pkexec ninja -C /tmp/libinput-config-upstream/build install
```

安装后会写入 `/usr/local/lib64/libinput-config.so` 和 `/etc/ld.so.preload`。

只配置滚动倍率，不覆盖自然滚动等 GNOME 设置：

```bash
pkexec sh -c 'cat > /etc/libinput.conf <<EOF
scroll-factor=0.15
EOF'
```

**生效方式**：注销并重新登录。Wayland/Mutter 需要重新启动会话才会加载新的 libinput 配置。

**调整速度**：数值越小越慢。

```bash
# 更慢
pkexec sh -c "sed -i 's/scroll-factor=.*/scroll-factor=0.10/' /etc/libinput.conf"
# 稍快
pkexec sh -c "sed -i 's/scroll-factor=.*/scroll-factor=0.20/' /etc/libinput.conf"
```

**验证**：

```bash
libinput list-devices
# 应看到：libinput-config: option 'scroll-factor' is '0.15'
```

**回滚**：

```bash
pkexec sh -c "rm -f /etc/libinput.conf && sed -i '/libinput-config.so/d' /etc/ld.so.preload"
pkexec rm -f /usr/local/lib64/libinput-config.so
```

## 桌面环境调教

### 系统语言与用户目录

系统界面设成中文，但用户路径保持英文。配置文件 `~/.config/user-dirs.dirs` 确保：

```bash
XDG_DESKTOP_DIR="$HOME/Desktop"
XDG_DOWNLOAD_DIR="$HOME/Downloads"
XDG_DOCUMENTS_DIR="$HOME/Documents"
XDG_PICTURES_DIR="$HOME/Pictures"
XDG_VIDEOS_DIR="$HOME/Videos"
```

### 显示桌面快捷键

GNOME 默认的 `Super+D`（显示桌面）是空的，需要手动绑定：

```bash
gsettings set org.gnome.desktop.wm.keybindings show-desktop "['<Super>d']"
```

### 桌面图标

GNOME 40+ 移除了桌面图标渲染功能，`~/Desktop/` 里有文件但背景上看不到。Fedora 仓库里也没有对应的 RPM 包，需要手动安装 Desktop Icons NG (DING) 扩展：

```bash
mkdir -p ~/.local/share/gnome-shell/extensions/ding@rastersoft.com
curl -sL "https://extensions.gnome.org/download-extension/ding@rastersoft.com.shell-extension.zip?version_tag=72024" -o /tmp/ding.zip
unzip -o /tmp/ding.zip -d ~/.local/share/gnome-shell/extensions/ding@rastersoft.com/
rm /tmp/ding.zip
```

注销重登录后桌面图标出现，拖放文件也正常了。

## 常用验证命令

```bash
google-chrome-stable --version
code --version
flatpak info md.obsidian.Obsidian
fcitx5-diagnose
gsettings get org.gnome.desktop.wm.keybindings show-desktop
gsettings get org.gnome.desktop.peripherals.touchpad speed
libinput list-devices
```

## Fedora 与主流 Linux 桌面对比

| 维度 | Fedora 44 | Ubuntu 25.04 | Arch Linux | openSUSE Tumbleweed | Linux Mint 22 |
|------|-----------|--------------|------------|---------------------|---------------|
| **包管理** | dnf5 + Flatpak | apt + Snap | pacman + AUR | zypper + Flatpak | apt + Flatpak |
| **默认桌面** | GNOME (vanilla) | GNOME (modified) | 自选 | KDE Plasma / GNOME | Cinnamon |
| **发布模型** | 6 个月一版，~13 月支持 | 6 个月一版，9 月支持 | 滚动更新 | 滚动更新 | 基于 Ubuntu LTS，5 年支持 |
| **内核** | 最新稳定版 | 略滞后 | 最新稳定版 | 最新稳定版 | LTS 内核 |
| **Wayland** | 默认启用 | 默认启用 | 取决于 DE | 默认启用 | 可选 |
| **开箱即用** | 中等 | 高 | 低（需自行安装配置） | 中高 | 高 |
| **软件新鲜度** | 高 | 中 | 最高 | 最高 | 低（LTS 稳定版） |
| **社区/文档** | 强（Fedora Wiki） | 最强（Ask Ubuntu） | 最强（Arch Wiki） | 中等 | 中等（继承 Ubuntu） |
| **适合场景** | 开发者、追新党 | 通用、新手友好 | 极客、高度定制 | 追新但求稳 | 从 Windows 迁移 |

> 选 Fedora 的核心原因是"上游优先"——新特性最早在 Fedora 落地，GNOME 体验也最接近上游设计意图。代价是偶尔会遇到兼容性问题，但对于开发用笔记本来说可以接受。
