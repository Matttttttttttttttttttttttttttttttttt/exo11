#!/bin/bash
read -p "entrez un commit " cmsg

if [ "$cmsg" == "" ]; then
  echo "MAIS IL FAUT UN MSG ENORME FDP GIGAFATASS"
else 
  echo ":)"
fi

git add . && git commit -m "$cmsg" && git push origin main 


