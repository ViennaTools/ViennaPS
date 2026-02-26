### Goal

Set up Docker on Ubuntu 24.04 with CUDA GPU support so containers can access the NVIDIA driver.

---

## 1) Install NVIDIA driver (host)

Check first:

```bash
nvidia-smi
```

If this works, skip. Otherwise install a driver:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-550
sudo reboot
```

Verify again:

```bash
nvidia-smi
```

---

## 2) Install Docker (official repo)

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Allow running docker without sudo:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Test:

```bash
docker run hello-world
```

---

## 3) Install NVIDIA Container Toolkit

Add repo and key:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo sed -i "s/\$(ARCH)/$(dpkg --print-architecture)/g" \
  /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Install toolkit:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Configure Docker runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 4) Verify GPU inside Docker

Run:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

You should see your GPU table printed from inside the container.

---

## 5) Usage example

Run an interactive CUDA container:

```bash
docker run --rm -it --gpus all nvidia/cuda:12.6.0-devel-ubuntu24.04 bash
```

Now CUDA programs inside the container use:

* CUDA toolkit from the image
* NVIDIA driver from the host

This is the required setup for running ViennaPS with GPU support inside Docker.
