#!/bin/sh

# This file has to be made executable in order to write the udev rules
# This can be done by opening a shell in the crappy/util folder and then running: 
# $ chmod 777 udev_rule_setter.sh 

# Once this is done, open a shell in the crappy/util folder and run:
# $ sudo ./udev_rule_setter.sh
# to actually set the udev rules


# Reaching the udev rules directory
cd ..
if [ -d "/etc/udev/rules.d" ]; then
  cd /etc/udev/rules.d
else
  echo "The udev rules directory (/etc/udev/rules.d) doesn't seem to exist, aborting";
  exit
fi

cd /etc/udev/rules.d


# The different rules that can be written
pololu="SUBSYSTEM==\"usb\", ATTR{idVendor}==\"1ffb\", MODE=\"0666\""
ft232h="SUBSYSTEM==\"usb\", ATTR{idVendor}==\"0403\", MODE=\"0666\""
seekth="SUBSYSTEM==\"usb\", ATTR{idVendor}==\"289d\", MODE=\"0777\""


# Choosing the rule to write, retrying if the input is not valid
echo "Which rule should be written ?"
echo "1. Pololu Tic"
echo "2. FT232H"
echo "3. Seek Thermal Pro"
echo "(ctrl + c to escape)"
echo ""

echo -n "Rule n°: "
read rule
echo ""

while { [ $rule -gt 3 ] || [ $rule -le 0 ]; } ; do
  echo "Invalid choice !"
  echo "Which rule should be written ?"
  echo ""
  echo -n "Rule n°: "
  read rule
  echo ""
done


# Writing the rule
case $rule in
        1)          echo "$pololu" > pololu.rules           ;;
        2)          echo "$ft232h" > ftdi.rules             ;;
        3)          echo "$seekth" > seek_thermal.rules     ;;
        *)          echo "Unexpected rule, aborting"; exit  ;;
esac


# Checking if the rule was actually written
case $rule in
        1)          if [ -f "pololu.rules" ]; then
                      echo "Writing successful !"
                    else
                      echo "Something went wrong, the rule is missing !"                 
                    fi                                                   ;;
        2)          if [ -f "ftdi.rules" ]; then
                      echo "Writing successful !"
                    else
                      echo "Something went wrong, the rule is missing !"                 
                    fi                                                   ;;
        3)          if [ -f "seek_thermal.rules" ]; then
                      echo "Writing successful !"
                    else
                      echo "Something went wrong, the rule is missing !"                 
                    fi                                                   ;;
esac


# Reloading the udev rules
udevadm control --reload-rules
if [ $? -eq 0 ]; then
  echo "";
  echo "The udev rules have been reloaded, please allow a few minutes before the change is effective."
else
  echo "";
  echo "Reloading of the udev rules from the command line was unsuccessful, rebooting will reload them."
fi

