while getopts c: flag
do
    case "${flag}" in
        c) config=${OPTARG};;
    esac
done
echo "USING CONFIG FILE: $config";
python draw_classifier/train.py --config $config && python draw_classifier/results.py --config $config 
