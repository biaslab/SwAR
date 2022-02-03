module SwarExperiments

# using ReactiveMP

# ## Hotfix for ReactiveMP and savename from DrWatson
# Base.string(meta::ImportanceSamplingApproximation) = "sampling($(meta.nsamples))"
# Base.string(::EM)              = "em"
# Base.string(::Marginalisation) = "vi"

include("swar_utils.jl")
include("swar_model.jl")

end