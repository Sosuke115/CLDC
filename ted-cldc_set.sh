for category in "art" "arts" "biology" "business" "creativity" "culture" "design" "economics" "education" "entertainment" "global" "health" "politics" "science" "technology";do
    echo $category  
    for mode in "train" "test";do
        echo $mode
        for lang in "en-fr" "fr-en";do
            echo $lang
            rm -r data/ted-cldc/"$lang"/"$mode"/"$category"/all
            mkdir data/ted-cldc/"$lang"/"$mode"/"$category"/all
            cat data/ted-cldc/"$lang"/"$mode"/"$category"/positive/* > data/ted-cldc/"$lang"/"$mode"/"$category"/all/positive.txt
            cat data/ted-cldc/"$lang"/"$mode"/"$category"/negative/* > data/ted-cldc/"$lang"/"$mode"/"$category"/all/negative.txt
            if [ $lang = "en-fr" ]; then
                sed -e 's/_en//g' data/ted-cldc/"$lang"/"$mode"/"$category"/all/./positive.txt > data/ted-cldc/"$lang"/"$mode"/"$category"/all/./positive.new.txt
                sed -e 's/_en//g' data/ted-cldc/"$lang"/"$mode"/"$category"/all/./negative.txt > data/ted-cldc/"$lang"/"$mode"/"$category"/all/./negative.new.txt 
            else   
                sed -e 's/_fr//g' data/ted-cldc/"$lang"/"$mode"/"$category"/all/./positive.txt > data/ted-cldc/"$lang"/"$mode"/"$category"/all/./positive.new.txt
                sed -e 's/_fr//g' data/ted-cldc/"$lang"/"$mode"/"$category"/all/./negative.txt > data/ted-cldc/"$lang"/"$mode"/"$category"/all/./negative.new.txt 
            fi     
        done     
    done
done

