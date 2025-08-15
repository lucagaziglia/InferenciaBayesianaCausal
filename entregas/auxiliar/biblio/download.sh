#!/bin/bash

#ssh address-of-B 'wget -O - http://server-C/whatever' >> whatever

BASEDIR=$(dirname "$0")

if [ $# \> 0 ]
then
    sed -n "/$1/,/^}/p" $BASEDIR/biblio.bib | grep note | grep -o '{.*}' | sed  's/{/"/g' | sed 's/}/"/g' | xargs wget -U --adjust-extension -O $BASEDIR/download/$1.pdf
else
    items=$(cat $BASEDIR/biblio.bib | grep @ | awk -F '{' '{print $2}' | awk -F ',' '{print $1}')
    for i in $items
    do
        if [ ! -f "$BASEDIR/download/$i" ]
	then
	    sed -n "/$i/,/^}/p" $BASEDIR/biblio.bib | grep note | grep -o '{.*}' | sed  's/{/"/g' | sed 's/}/"/g' | xargs wget -U --adjust-extension -O $BASEDIR/download/$i.pdf
        fi
    done
    echo "All biblio download"
fi



