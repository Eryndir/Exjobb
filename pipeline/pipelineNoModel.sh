decades=( 2010 2000 1990 1980 1970 1960 1950 1940 1930 )

#echo "Starting script on dialectic names"
#for decade in "${decades[@]}"; do
#    /home/tmpuser/code/Exjobb-main/venv/bin/python /home/tmpuser/code/Exjobb-main/pipeline/pipeline.py "$decade" 1 0
#done

echo "Starting script on regular names, saving index"
for decade in "${decades[@]}"; do
    /home/tmpuser/code/Exjobb-main/venv/bin/python /home/tmpuser/code/Exjobb-main/pipeline/pipeline.py "$decade" 0 1 0
done

#echo "Starting script on other words, soon...."

#for decade in "${decades2[@]}"; do
#    /home/tmpuser/code/Exjobb-main/venv/bin/python /home/tmpuser/code/Exjobb-main/pipeline/pipeline.py "$decade" 2 0
#done
