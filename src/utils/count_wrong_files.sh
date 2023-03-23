for FILE in /data/Coronavirus-Tweets/Covid_Concerns/mf_annotations/*.csv
do
    tmp=$(tail -n 1 $FILE | grep -c ",,")
    if (($tmp > 0))
    then
        echo $FILE
    fi
done

echo "done"