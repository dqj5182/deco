#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# username and password input
echo -e "\nYou need to register at https://rich.is.tue.mpg.de/"
read -p "Username:" username
read -p "Password:" password
read -p "Save directory:" save_dir
username=$(urle $username)
password=$(urle $password)
save_dir=$save_dir

mkdir $save_dir

# Download
# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=test_hsc.zip' -O $save_dir'/test_hsc.zip' -P save_dir --no-check-certificate --continue 
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=val_body.zip' -O $save_dir'/val_body.zip' -P save_dir --no-check-certificate --continue 