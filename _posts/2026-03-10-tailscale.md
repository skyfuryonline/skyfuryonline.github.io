---
layout: post
title: "TailScale"
date: 2026-03-10
author: "LH"
catalog: true
tags: [tailscale]
group: vision
---

## 引言

学校服务器或家中主机虽然方便，但是由于网络限制，无法使用外面的WIFI或移动网络直接连接使用。基于这一限制条件，补充一种可以安全地将不同设备连接到同一个私有网络的工具：tailscale。

![tailscale](/img/vision/tailscale/tailscale.png)

tailscale介绍如下：
- Tailscale 是一款基于 WireGuard 协议构建的现代化 Mesh VPN（网状虚拟专用网络） 工具，用于安全地将不同设备连接到同一个私有网络。与传统 VPN 需要集中服务器不同，Tailscale 采用点对点（P2P）连接架构，使设备之间可以直接建立加密通信，从而降低延迟并提升连接效率。
- 具有以下优点：
  - **零配置组网**：只需登录账号即可自动加入同一网络（称为 Tailnet），无需手动配置端口转发或复杂的网络规则；
  - **跨平台支持**：支持 Windows、macOS、Linux、iOS、Android，以及路由器、NAS、容器等环境；
  - **基于身份的访问控制**：通过 OAuth 或 SSO（如 Google、Microsoft、GitHub 等）管理设备和用户权限，实现零信任网络（Zero Trust）。
  - **端到端加密**：所有通信使用 WireGuard 加密，只有参与通信的设备能够解密数据。
  - **内网穿透能力**：即使设备位于 NAT 或防火墙后，也能通过 NAT traversal 技术建立连接。

[tailscale](https://tailscale.com/)


## 配置流程

选择特定的安装包:    
```bash
cd ~
wget https://pkgs.tailscale.com/stable/tailscale_1.94.2_amd64.tgz   

tar xzf tailscale_*.tgz
mv tailscale_*/ tailscale/
rm tailscale_*.tgz
```

设置PATH 和快捷别名:    
```bash
cat >> ~/.bashrc << 'EOF'

export PATH="$HOME/tailscale:$PATH"
alias tailscale='tailscale --socket=$HOME/tailscale/tailscaled.sock'
EOF

source ~/.bashrc
```

创建工作路径：  
```bash
mkdir -p ~/tailscale
```

持久化：    
```bash
cat > ~/tailscale/start_tailscaled.sh << 'EOF'  
#!/bin/bash 
cd ~  
TAILSCALED="$HOME/tailscale/tailscaled --state=$HOME/tailscale/tailscaled.state --socket=$HOME/tailscale/tailscaled.sock --tun=userspace-networking --port=41641" 
if ! pgrep -x tailscaled > /dev/null; then  
    nohup $HOME/tailscale/tailscaled --cleanup >> ~/tailscale/tailscaled.log 2>&1 & 
    nohup $TAILSCALED >> ~/tailscale/tailscaled.log 2>&1 &  
    echo "$(date) tailscaled 已启动" >> ~/tailscale/tailscaled.log  
fi  
EOF 

chmod +x ~/tailscale/start_tailscaled.sh  
(crontab -l 2>/dev/null; echo "@reboot $HOME/tailscale/start_tailscaled.sh"; echo "* * * * * $HOME/tailscale/start_tailscaled.sh") | crontab -
$HOME/tailscale/start_tailscaled.sh 
```


## 登陆组网

[控制台](https://login.tailscale.com/admin/machines)    
```bash
tailscale up --auth-key=your-api-key
```
> 其中`api-key`在`Personal Settings/keys`中手动生成`Auth keys`即可；

验证并查看分配的IP：    
```bash
tailscale status
tailscale ip -4          
# 查看你的 Tailscale IP
```


