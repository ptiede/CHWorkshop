using Pkg;Pkg.activate(@__DIR__)
using VLBIFiles
using CairoMakie
using Comrade
using FillArrays
using AdvancedHMC
using Optimization
using OptimizationOptimisers
using Random
using Distributions
using VLBIImagePriors
using Enzyme
using Plots
using CSV
using Printf
using Serialization
using BenchmarkTools
using LinearAlgebra
using LogDensityProblems
using FINUFFT
LinearAlgebra.BLAS.set_num_threads(1)
# Comrade.VLBISkyModels.NFFT._use_threads[] = false
# Comrade.VLBISkyModels.FFTW.set_num_threads(1)


include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "instrumentmodels.jl"))
include(joinpath(@__DIR__, "skymodels.jl"))


function comrade_imager(
        outbase::String, skym, intm, data...; maxiters = 15_000, ntrials = 10,
        nsample = 10_000, nadapt = 5_000, rng = Random.default_rng(), restart = false,
        benchmark = true, start = nothing, imgdata = nothing
    )

    @info outbase
    mkpath(dirname(outbase))
    outimg = mkpath(joinpath(dirname(outbase), "images"))
    @info "Outputing to $outbase"


    post = VLBIPosterior(skym, intm, data...; imgdata)
    tpost = asflat(post)
    ndim = dimension(tpost)

    if benchmark
        @info "Forward Pass benchmark"
        x0 = randn(ndim)
        btt = @benchmark logdensityof($tpost, $x0)
        io = IOContext(stdout)
        show(io, MIME("text/plain"), btt)
        println()

        @info "Reverse Pass benchmark"
        btt = @benchmark Comrade.LogDensityProblems.logdensity_and_gradient($(tpost), $x0)
        io = IOContext(stdout)
        show(io, MIME("text/plain"), btt)
        println()

    end

    out = outbase
    g = post.skymodel.grid.imgdomain
    # refine the grid for plotting
    gimg = refinespatial(g, 2)


    if !restart && isnothing(start)
        post0 = VLBIPosterior(skym, intm, add_fractional_noise(data[1], 0.05); imgdata)
        sols, ℓopt = best_image(post0, ntrials, maxiters, rng)

        post1 = VLBIPosterior(skym, intm, add_fractional_noise(data[1], 0.025); imgdata)
        xopt1, sol = comrade_opt(
            post1, Adam();
            initial_params = sols[1], maxiters = maxiters ÷ 2, g_tol = 1.0e-1
        )

        residual(post1, xopt1)
        savefig(out * "_residuals_step1_map.png")


        post2 = VLBIPosterior(skym, intm, add_fractional_noise(data[1], 0.01); imgdata)
        xopt2, sol = comrade_opt(
            post2, Adam();
            initial_params = xopt1, maxiters = maxiters ÷ 2, g_tol = 1.0e-1
        )

        residual(post2, xopt2)
        savefig(out * "_residuals_step2_map.png")

        xopt, sol = comrade_opt(
            post, Adam();
            maxiters = maxiters, g_tol = 1.0e-1,
            initial_params = xopt2
        )

        img = intensitymap(skymodel(post, xopt), gimg)

        Comrade.save_fits(out * "_optimal.fits", img)
        p = imageviz(img)
        CairoMakie.save(out * "_optimal.png", p)
        residual(post, xopt)
        savefig(out * "_residuals_final_map.png")


        if hasproperty(xopt, :instrument)
            k = keys(xopt.instrument)
            v = values(xopt.instrument)
            map(k, v) do ki, vi
                gtp = Comrade.caltable(vi)
                CSV.write(out * "_ctable_$ki.csv", gtp)
                Plots.plot(gtp, layout = (4, 4), size = (800, 500))
                savefig(out * "_ctable_$ki.png")
            end
        end

        serialize(
            out * "_optimum_allres.jls", Dict(
                :xopt => xopt,
                :post => post
            )
        )
    elseif !restart && !isnothing(start)
        @info "Starting from passed location. The logdensity is $(logdensityof(post, start))"
        xopt = start

        img = intensitymap(skymodel(post, xopt), gimg)

        save_fits(out * "_start.fits", img)
        p = imageviz(img)
        CairoMakie.save(out * "_start.png", p)

        residual(post, xopt)
        savefig(out * "_residuals_map.png")

        if hasproperty(xopt, :instrument)
            k = keys(xopt.instrument)
            v = values(xopt.instrument)
            map(k, v) do ki, vi
                gtp = Comrade.caltable(vi)
                CSV.write(out * "_ctable_$ki.csv", gtp)
                Plots.plot(gtp, layout = (4, 4), size = (800, 500))
                savefig(out * "_ctable_$ki.png")
            end
        end

        serialize(
            out * "_optimum_allres.jls", Dict(
                :xopt => xopt,
                :post => post
            )
        )

    else
        xopt = deserialize(out * "_optimum_allres.jls")[:xopt]
    end

    integrator = Leapfrog(0.01)
    metric = DiagEuclideanMetric(dimension(tpost))
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(
        MassMatrixAdaptor(metric), StepSizeAdaptor(0.9, integrator);
        init_buffer = 200, term_buffer = 500
    )
    smplr = HMCSampler(kernel, metric, adaptor)

    trace = sample(rng, post, smplr, nsample; saveto = DiskStore(mkpath(out), 25), n_adapts = nadapt, initial_params = xopt, restart)
    @info nadapt
    if restart
        chain = load_samples(mkpath(out), (nadapt + 1):10:nsample)
    else
        chain = load_samples(trace, (nadapt + 1):10:nsample)
    end

    p = Plots.plot()
    ss = sample(chain, 10)
    p = residual(post, ss[begin])
    for s in ss[2:end]
        residual!(p, post, s)
    end
    savefig(out * "_residuals.png")


    @info "Saving images"
    samples = skymodel.(Ref(post), sample(chain, 500))
    for (i, s) in enumerate(samples)
        out = joinpath(outimg, basename(outbase) * @sprintf("draw_%03d.fits", i))
        save_fits(out, intensitymap(s, gimg))
    end
    return
end
