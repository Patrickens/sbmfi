automated installation (recommended):
install gdebi (sudo apt install gdebi)
After that install 13cflux-software package and all its dependencies
sudo gdebi x3cflux_2.1-4_amd64.deb

You may also want to install the extensive documentation available in the doc package.

manual installation (not recommended):
Install the packages in the sofware list.
It should be possible to install all packages with the packageinstaller.
After this install the 13cflux-Sofware-package as root:
dpkg -i x3cflux_2.1-4_amd64.deb 
