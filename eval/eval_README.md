cd /path/to/pysot-toolkit
python bin/eval.py \
	--dataset_dir pyCFTrackers/dataset \		# dataset path
	--dataset UAV123 \				# dataset name(OTB100, UAV123, NFS, LaSOT)
	--tracker_result_dir pyCFTrackers/cftracker \	# tracker dir
	--num 4 \				  	# evaluation thread
	--show_video_level \ 	  			# wether to show video results
	--vis 