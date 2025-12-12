## Create target folder on RPi

```bash
mkdir ~/py-frame
```

## Mount NAS photo folder using NFS

### Enable and configure NFS on NAS

* Login to NAS web UI and navigate to `Control Panel - File Services - NFS`.
* Enable NFS service and select NFSv3 (it might work with up to v4.1, but I did not try).
* Keep Advanced Settings as they are

![NFS Settings](./pictures/nfs.jpg)

### Configure Shared Folder NFS Permissions

* Click Shared Folder on the NFS settings page.
* Right-click on `photo` folder, choose `Edit` and go to `NFS Permissions`.
* Create a rule:
  * Hostname or IP: 192.168.1.201 (also try using `rpi` host name) -
     this is your RPi's IP.
  * Privilege: Read only
  * Squash: Map all users to admin (admin must have access to `photo`; see below)
  * Security: sys
  * Enable asynchronous
  * The rest of the settings might be not needed, but I set them too. 

![NFS Permissions](./pictures/nfs-perms.jpg)

### Make sure `admin` can access the shared folder

Since all incoming user names (like `pi`) are mapped/squashed to nasus' `admin`,
the `admin` must have access to photo folder:

![Admin Permissions](./pictures/admin-perms.jpg)

### Mount NAS NFS

#### First, try it manually
On `RPi` run:
```
sudo mount -t nfs -o vers=3 nasus:/volume1/photo /mnt/nasus/photo
```
Check that NFS  mount works and files are visible:
```bash
ls /mnt/nasus/photo
```
You should see photo's content and no errors.

#### Auto-mount NAS photo folder on RPi boot
```
sudo nano /etc/fstab
```
Add the following line:
```
nasus:/volume1/photo  /mnt/nasus/photo  nfs  vers=3,noauto,x-systemd.automount,_netdev,nofail,defaults,noatime,nolock,tcp,soft,timeo=50,retrans=2  0  0
```

Reboot RPi and check if you can still list photo files.

### Link mounted NAS folder to the target folder

The photos are listed as relative paths starting with `nasus/photo/...` in
both `photo.xxx.list` and `exclusions.txt`. This means that we need to map `nasus`
folder inside the target folder.

```bash
ln -s /mnt/nasus ~/py-frame/nasus
```

Check that it is mapped correctly:

```
ls -l ~/py-frame
```

You should see:
```
lrwxrwxrwx 1 pi pi      10 Dec 12 16:13 nasus -> /mnt/nasus
```

Check that photos are accessible:
```
ls -l ~/py-frame/nasus/photo
```

You should see something like:
```
total 1264
drwxrwxrwx  33 1026 users   4096 Mar 16  2023  2002
drwxrwxrwx 133 1026 users  12288 Dec  7  2024  2003
drwxrwxrwx 105 1026 users   4096 May 18  2022  2004
drwxrwxrwx  69 1026 users   4096 Mar 16  2023  2005
```

## Make slideshow start on RPi boot

### Step 1
Create file py-frame.service:

```bash
sudo nano /etc/systemd/system/py-frame.service
```

with this content:

```bash
[Unit]
Description=Raspberry Pi Photo Frame Slideshow (console)
After=network.target local-fs.target
Conflicts=getty@tty1.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/py-frame
ExecStart=/usr/bin/python /home/pi/py-frame/py_frame.py /home/pi/py-frame/photo.irina.list

# Bind the service to the main console (tty1)
StandardInput=tty
StandardOutput=tty
StandardError=journal
TTYPath=/dev/tty1
TTYReset=yes
TTYVHangup=yes

# Tell SDL/pygame to use the framebuffer
Environment=SDL_VIDEODRIVER=fbcon
Environment=SDL_FBDEV=/dev/fb0
Environment=SDL_NOMOUSE=1

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Step 2
Disable TTY1:

```bash
sudo systemctl disable getty@tty1.service
```

Reboot to check if slideshow starts on boot.

```bash
reboot
```





