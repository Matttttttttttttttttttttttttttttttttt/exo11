#!/bin/bash
read -p "entrez un commit " cmsg

if [ "$cmsg" == "" ]; then
  echo "MAIS IL FAUT UN MSG ENORME FDP GIGAFATASS"
else 
  echo ":)"
fi

git add . && git commit -m "$cmsg" && git push origin main 

if [ ! -d ".git"=0 ]; then
  echo "Erreur : Ce répertoire n'est pas un dépôt Git."
else 
  echo "ok"
fi
