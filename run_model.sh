while getopts c: flag
do
    case "${flag}" in
        c) config=${OPTARG};;
    esac
done
echo "USING CONFIG FILE: $config";
python train.py --config $config && python results.py --config $config 
