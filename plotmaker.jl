#plotmaker.jl
#I'm doing this in julia because it's easier than in Python
using PGFPlots, DataFrames, CSV, RollingFunctions

df = DataFrame(CSV.File("biped_ars00001.csv"))
name_array = names(df)
# temp = Matrix(df)
# samples = I*temp'
rollouts = Matrix(df)
episodes = rollouts[:,1]
reward = rollouts[:,2]
reward_std = rollouts[:,5]
moving_avg = 200
smoothed_reward = rolling(sum, reward, moving_avg)./moving_avg

p = Axis([
    Plots.Scatter(episodes, reward, legendentry="Rollout Data", markSize=0.1),
    Plots.Linear(episodes[moving_avg:end], smoothed_reward, legendentry="Average Reward", mark="none")
    ], xlabel="Episode", ylabel="Reward", title="Accumulated Reward vs Episode"
    )
# savefig("plots/biped_ars001.pdf")
# a = Axis(Plots.Linear(x, y, legendentry="My Plot"), xlabel="X", ylabel="Y", title="My Title")
# p = Axis(Plots.Linear(episodes, rollouts, legendentry="reward"))#, markSize=0.1, onlyMarks=true)
# p = Axis(Plots.Linear(episodes, rollouts, legendentry="reward", markSize=0.1, onlyMarks=true), xlabel="Episode", ylabel="Reward", title="Episode Reward vs Episode Number")
p.legendStyle = "at={(1.05,1.0)}, anchor=north west"
save("plots/biped_ars00001.pdf", p)



