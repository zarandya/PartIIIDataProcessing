#! /bin/zsh

pi_if=none

nmcli device show wlp7s0 | grep "piwifi"
if [ "$?" -eq "0" ]
then
	pi_if=wlp7s0
	server_ip=$(nmcli device show wlp7s0 | grep "IP4.ADDRESS\[1\]" | sed "s/^.*: *\\([0-9\\.]*\\)\\/.*$/\\1/")
	pi_ip=192.168.4.1
	pi_server_ip=192.168.4.1
fi

nmcli device show enp8s0 | grep "Ethernet sharing"
if [ "$?" -eq "0" ]
then
	pi_if=enp8s0
	server_ip=10.42.0.1
	pi_ip=10.42.0.241
	pi_server_ip=192.168.4.1
fi

if [ "$pi_if" = "none" ]
then
	echo Pi not found
	exit 1
fi


files=$(java -jar ./DataCollection/build/classes/artifacts/ComputerServer_jar/ComputerServer.jar $server_ip $pi_server_ip)

echo
echo Finished reading files:
echo "$files"
echo

M=0
rm proc0
rm proc1
rm proc2
rm proc3

while read line
do
	if [[ "$line" == *.json ]]
	then
		grep recordingRemoteFileName "$line"
		if [ "$?" -eq "0" ]
		then
			filename="$(sed "s/^.*\"recordingRemoteFileName\":\"\\([^\"]*\\)\".*$/\\1/" "$line")"
			echo "$filename"
			local_filename="${filename##*/}"
			local_dir="${line%/*}"
			rsync -avzz --remove-source-files "pi@$pi_ip:/home/pi/$filename" "$local_dir/$local_filename"
			#./sync.py "$line" > /dev/null
			echo "./sync.py \"$line\"" >> proc$M
			M=$(( (M+1) % 4))
		fi
	fi
done < <(echo $files)

bash proc0 &
bash proc1 &
bash proc2 &
bash proc3 &

