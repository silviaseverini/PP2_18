for ng in 1 3
do
  for sws in 7 9 11 13
  do
    for es in 64 100 256
      do
        for ra in 0.0001 0.001 0
        do
          for lr in 0.001 0.0001
          do
            for dp in 0.5 0.75 1
            do
              for bs in 32 64
              do 
                python param_search.py -ng $ng -sws $sws -es $es -ra $ra -lr $lr -dp $dp -bs $bs
            done
          done
        done
      done
    done
  done
done