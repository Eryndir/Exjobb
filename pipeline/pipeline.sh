decades=(1920 1930 1940 1950 1960 1970 1980 1990 2000 2010)

for decade in "${decades[@]}"; do
    /home/tmpuser/code/Exjobb-main/venv/bin/python /home/tmpuser/code/Exjobb-main/pipeline/pipeline.py "$decade"
done