To do :
Create a "x3cflux" folder in /etc .
Save license.dat in this directory:
/etc/x3cflux/license.dat

To use the applications of 13Cflux (except fmllint) you have to sign your fml file with fmlsign:

fmlsign -i "old fml file" -o "new fml file"

Than the fml file is sent to our server by https, it is sign there and send back to your computer.