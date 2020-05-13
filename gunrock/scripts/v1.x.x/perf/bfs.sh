# ------------------------------------------------------------------------
#  Gunrock (v1.x.x): Performance Testing Script(s)
# ------------------------------------------------------------------------

#!/bin/bash

EXEDIR=${1:-"../../../build/bin"}
DATADIR=${2:-"/data/gunrock_dataset/large"}
DEVICE=${3:-"0"}
TAG=${4:-""}

APPLICATION="bfs"
EXECUTION="$EXEDIR/$APPLICATION"

ORG_OPTIONS=""
ORG_OPTIONS="$ORG_OPTIONS --num-runs=10"
# ORG_OPTIONS="$ORG_OPTIONS --validation=each"
ORG_OPTIONS="$ORG_OPTIONS --quick"
ORG_OPTIONS="$ORG_OPTIONS --device=$DEVICE"
ORG_OPTIONS="$ORG_OPTIONS --src=random"
ORG_OPTIONS="$ORG_OPTIONS --64bit-SizeT=false,true"
ORG_OPTIONS="$ORG_OPTIONS --64bit-VertexT=false,true"
ORG_OPTIONS="$ORG_OPTIONS --idempotence=false,true"
ORG_OPTIONS="$ORG_OPTIONS --mark-pred=false,true"
ORG_OPTIONS="$ORG_OPTIONS --direction-optimized=true,false"
ORG_OPTIONS="$ORG_OPTIONS --advance-mode=LB_CULL,LB,TWC"
ORG_OPTIONS="$ORG_OPTIONS --tag=$TAG"

EVAL_DIR="BFS"

NAME[ 0]="ak2010"            &&  OPT_DIR[ 0]=""

NAME[ 1]="delaunay_n10"      && OPT_UDIR[ 1]="--do-a=0.1      --do-b=-1" 
NAME[ 2]="delaunay_n11"      && OPT_UDIR[ 2]="--do-a=0.1      --do-b=-1" 
NAME[ 3]="delaunay_n12"      && OPT_UDIR[ 3]="--do-a=0.1      --do-b=-1" 
NAME[ 4]="delaunay_n13"      && OPT_UDIR[ 4]="--do-a=0.1      --do-b=-1" 
NAME[ 5]="delaunay_n14"      && OPT_UDIR[ 5]="--do-a=0.1      --do-b=-1" 
NAME[ 6]="delaunay_n15"      && OPT_UDIR[ 6]="--do-a=0.1      --do-b=-1" 
NAME[ 7]="delaunay_n16"      && OPT_UDIR[ 7]="--do-a=0.1      --do-b=-1" 
NAME[ 8]="delaunay_n17"      && OPT_UDIR[ 8]="--do-a=0.1      --do-b=-1" 
NAME[ 9]="delaunay_n18"      && OPT_UDIR[ 9]="--do-a=0.1      --do-b=-1" 
NAME[10]="delaunay_n19"      && OPT_UDIR[10]="--do-a=0.1      --do-b=-1" 
NAME[11]="delaunay_n20"      && OPT_UDIR[11]="--do-a=0.1      --do-b=-1" 
NAME[12]="delaunay_n21"      && OPT_UDIR[12]="--do-a=0.1      --do-b=-1" 
NAME[13]="delaunay_n22"      && OPT_UDIR[13]="--do-a=0.1      --do-b=-1" 
NAME[14]="delaunay_n23"      && OPT_UDIR[14]="--do-a=0.1      --do-b=-1" 
NAME[15]="delaunay_n24"      && OPT_UDIR[15]="--do-a=0.1      --do-b=-1" 

NAME[21]="kron_g500-logn16"  && OPT_UDIR[21]="--do-a=0.00002  --do-b=0.01" 
NAME[22]="kron_g500-logn17"  && OPT_UDIR[22]="--do-a=0.00002  --do-b=0.01" 
NAME[23]="kron_g500-logn18"  && OPT_UDIR[23]="--do-a=0.00002  --do-b=0.01" 
NAME[24]="kron_g500-logn19"  && OPT_UDIR[24]="--do-a=0.00002  --do-b=0.01" 
NAME[25]="kron_g500-logn20"  && OPT_UDIR[25]="--do-a=0.00002  --do-b=0.01"  && I_SIZE_UDIR[20]="1.50" 
NAME[26]="kron_g500-logn21"  && OPT_UDIR[26]="--do-a=0.00002  --do-b=0.01"
NAME[27]="rmat_n24_e16"      &&    GRAPH[27]="rmat --graph-scale=24 --graph-edgefactor=16"
                                 OPT_DIR[27]="--do-a=0.000005 --do-b=0.5"
                                OPT_UDIR[27]="--do-a=0.000002 --do-b=0.2"
NAME[28]="rmat_n23_e32"      &&    GRAPH[28]="rmat --graph-scale=23 --graph-edgefactor=32"
                                 OPT_DIR[28]="--do-a=0.000005 --do-b=2"
                                OPT_UDIR[28]="--do-a=0.00001  --do-b=1"
NAME[29]="rmat_n22_e64"      &&    GRAPH[29]="rmat --graph-scale=22 --graph-edgefactor=64"
                                 OPT_DIR[29]="--do-a=0.00001  --do-b=0.1"
                                OPT_UDIR[29]="--do-a=0.000005 --do-b=1"

NAME[31]="coAuthorsDBLP"     && OPT_UDIR[31]="--do-a=0.005    --do-b=0.00002"
NAME[32]="coAuthorsCiteseer" && OPT_UDIR[32]="--do-a=0.02     --do-b=0.0002" 
NAME[33]="coPapersDBLP"      && OPT_UDIR[33]="--do-a=1.0      --do-b=0.001"  
NAME[34]="coPapersCiteseer"  && OPT_UDIR[34]="--do-a=0.5      --do-b=0.001"  
NAME[35]="citationCiteseer"  && OPT_UDIR[35]="--do-a=0.5      --do-b=2"      
NAME[36]="preferentialAttachment" && OPT_UDIR[36]="--do-a=0.0005 --do-b=0.000005" 

NAME[41]="soc-LiveJournal1"  &&  OPT_DIR[41]="--do-a=0.05     --do-b=2"      
                                OPT_UDIR[41]="--do-a=0.02     --do-b=0.0001" 
NAME[42]="soc-twitter-2010"  && OPT_UDIR[42]="--do-a=0.005    --do-b=0.1     --queue-factor=1.1"
NAME[43]="soc-orkut"         && OPT_UDIR[43]="--do-a=5        --do-b=10 "
NAME[44]="hollywood-2009"    && OPT_UDIR[44]="--do-a=0.01     --do-b=0.0005"
NAME[45]="soc-sinaweibo"     &&  OPT_DIR[45]="--do-a=0.00005  --do-b=0.00005 --queue-factor=1.1"
                                OPT_UDIR[45]="--do-a=0.00005  --do-b=0.0001  --queue-factor=1.1"

NAME[51]="webbase-1M"        &&  OPT_DIR[51]="--do-a=100      --do-b=0.01"
                                OPT_UDIR[51]="--do-a=100      --do-b=0.1"    
NAME[52]="arabic-2005"       &&  OPT_DIR[52]="--do-a=500      --do-b=10    --queue-factor=1.1"
                                OPT_UDIR[52]="--do-a=500      --do-b=20    --queue-factor=1.1"
NAME[53]="uk-2002"           &&  OPT_DIR[53]="--do-a=20000    --do-b=0.05"
                                OPT_UDIR[53]="--do-a=20       --do-b=200"
NAME[54]="uk-2005"           &&  OPT_DIR[54]="--do-a=200      --do-b=2000  --queue-factor=1.1"
                                OPT_UDIR[54]="--do-a=20       --do-b=2     --queue-factor=1.1"
NAME[55]="webbase-2001"      &&  OPT_DIR[55]="--do-a=50       --do-b=0.002 --queue-factor=2"
                                OPT_UDIR[55]="--do-a=5        --do-b=20    --queue-factor=2"
NAME[56]="indochina-2004"    &&  OPT_DIR[56]="--do-a=1000     --do-b=0.001"
                                OPT_UDIR[56]="--do-a=200      --do-b=1000"
NAME[57]="caidaRouterLevel"  && OPT_UDIR[57]="--do-a=0.1      --do-b=0.00005" 

NAME[61]="roadNet-CA"        && OPT_UDIR[61]="--do-a=-1.0                  --queue-factor=0.01" 
NAME[62]="belgium_osm"       && OPT_UDIR[62]="--do-a=-1.0                  --queue-factor=0.01" 
NAME[63]="netherlands_osm"   && OPT_UDIR[63]="--do-a=-1.0                  --queue-factor=0.01" 
NAME[64]="italy_osm"         && OPT_UDIR[64]="--do-a=-1.0                  --queue-factor=0.01" 
NAME[65]="luxembourg_osm"    && OPT_UDIR[65]="--do-a=-1.0                  --queue-factor=0.01" 
NAME[66]="great-britain_osm" && OPT_UDIR[66]="--do-a=-1.0                  --queue-factor=0.01" 
NAME[67]="germany_osm"       && OPT_UDIR[67]="--do-a=0.05     --do-b=-1    --queue-factor=0.01" 
NAME[68]="asia_osm"          && OPT_UDIR[68]="--do-a=-1.0                  --queue-factor=0.02" 
NAME[69]="europe_osm"        && OPT_UDIR[69]="--do-a=2.5      --do-b=0.001 --queue-factor=0.02" 
NAME[70]="road_usa"          && OPT_UDIR[70]="--do-a=-1.0                  --queue-factor=0.01" 
NAME[71]="road_central"      && OPT_UDIR[71]="--do-a=0.1      --do-b=0.00001 --queue-factor=0.01"

#NAME[41]="tweets"            && OPT_DIR[41]="--queue-factor=5.00" && I_SIZE_DIR[41]="2.00" && OPT_UDIR[41]="--queue-factor=5.00" && I_SIZE_UDIR[41]="2.00" 
#NAME[42]="bitcoin"           && OPT_DIR[42]="--queue-factor=5.00" && I_SIZE_DIR[42]="2.00" && OPT_UDIR[42]="--queue-factor=10.0" && I_SIZE_UDIR[42]="2.00" 

mkdir -p $EVAL_DIR

for i in {0..71}; do
    if [ "${NAME[$i]}" = "" ]; then
        continue
    fi

    SUFFIX=${NAME[$i]}
    mkdir -p $EVAL_DIR/$SUFFIX

    for undirected in "true" "false"; do
        OPTIONS=$ORG_OPTIONS
        MARKS=""

        if [ "${GRAPH[$i]}" = "" ]; then
            GRAPH_="market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx"
        else
            GRAPH_="${GRAPH[$i]}"
        fi

        if [ "$undirected" = "true" ]; then
            OPTIONS_="$OPTIONS ${OPT_UDIR[$i]} --undirected=true"
            MARKS="UDIR"
        else
            OPTIONS_="$OPTIONS ${OPT_DIR[$i]}"
            if [ "${OPT_DIR[$i]}" = "" ]; then
                continue
            fi
            MARKS="DIR"
        fi

        echo $EXECUTION $GRAPH_ $OPTIONS_ --jsondir=./$EVAL_DIR/$SUFFIX "> ./$EVAL_DIR/$SUFFIX/${NAME[$i]}.${MARKS}.txt"
             $EXECUTION $GRAPH_ $OPTIONS_ --jsondir=./$EVAL_DIR/$SUFFIX  > ./$EVAL_DIR/$SUFFIX/${NAME[$i]}.${MARKS}.txt
        sleep 1
    done
done
