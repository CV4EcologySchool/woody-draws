sbatch --mem 4000 -J rclonecopy -t 5-00:00:00 -o backup.out -p griz_partition_gpu anybatch -2 rclone copy remote:WoodyDraws/ data/images/
