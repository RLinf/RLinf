tensorboard --logdir logs --port 6007 --host 0.0.0.0 

# 当前的是metric + 合并最新rlinf，但是有bug，
# 因此metric仓库验证metric没问题，before_rstar2仓库验证metric+ rstar2 commit前没问题