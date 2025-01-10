#!/bin/bash
read -p "entrez un commit" cmsg

git add . && git commit -m "$cmsg" && git push origin main 

