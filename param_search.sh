n_grams=(1 3 5)
sliding_window_sizes=(5 7 9 11 13 15)
embedding_sizes=(100 256 512)
reg_alphas=(0.0001 0.001 0.01 0.0005)
lrs=(0.0001 0.001 0.01 0.0005)
dropout_probs=(0.5 0.6 0.7 0.75)
batch_sizes=(32 64 128)

for ng in 1 3 5
do
  for sws in 13 15
  do
    for es in 100 256 512
      do
        for ra in 0.0001 0.001 0.01 0.0005
        do
          for lr in 0.0001 0.001 0.01 0.0005
          do
            for dp in 0.5 0.6 0.7 0.75
            do
              for bs in 32 64 128
                do 
                  # echo $ra
                  # echo "${ng} ${sws} ${es} ${ra} ${lr} ${dp} ${bs}";
                  python param_search.py -ng $ng -sws $sws -es $es -ra $ra -lr $lr -dp $dp -bs $bs
            done
          done
        done
      done
    done
  done
done