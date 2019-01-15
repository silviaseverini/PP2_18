for ng in 1 3 5
do
  for sws in 7 9 11 13
  do
    for es in 64 100 256
      do
        for ra in 0.0001 0.001
        do
          for lr in 0.0001 0.001
          do
            for dp in 0.6 0.75
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