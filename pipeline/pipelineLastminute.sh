decades=(1980)

#echo "Starting script on dialectic names"
#for decade in "${decades[@]}"; do
#    /home/tmpuser/code/Exjobb-main/venv/bin/python /home/tmpuser/code/Exjobb-main/pipeline/pipeline.py "$decade" 1 0
#done

echo "Starting script on regular names, saving index and no bertopic model"
for decade in "${decades[@]}"; do
    echo "${decade}"
    /home/tmpuser/code/Exjobb-main/venv/bin/python /home/tmpuser/code/Exjobb-main/pipeline/pipeline.py "$decade" 0 1 0
done

#echo "Starting script on other words, soon...."

#for decade in "${decades2[@]}"; do
#    /home/tmpuser/code/Exjobb-main/venv/bin/python /home/tmpuser/code/Exjobb-main/pipeline/pipeline.py "$decade" 2 0
#done
