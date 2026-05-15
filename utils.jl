using OptimizationOptimisers

function load_chain_and_post(file, nsamples = Base.Colon())
    chain = load_samples(file, nsamples)
    post = deserialize(file * "_optimum_allres.jls")[:post]
    return chain, post
end

using Printf
function saveimgs(imgs, outpath)
    for (i, img) in enumerate(imgs)
        save_fits(joinpath(outpath, @sprintf("draw_img_%03d.fits", i)), img)
    end
end

mutable struct FCallback{F}
    counter::Int
    stride::Int
    const f::F
end
FCallback(stride, f) = FCallback(0, stride, f)
function (c::FCallback)(x, others)
    c.counter += 1
    if c.counter % c.stride == 0
        @info "On step $(c.counter) f = $(x.objective)"
        return false
    else
        return false
    end
end


function best_image(post, ntrials = 20, maxiters = 10_000, rng = Random.default_rng())
    nd = mapreduce(Comrade.ndata, +, post.data)
    sols = map(1:ntrials) do i
        xopt0, sol0 = comrade_opt(
            post, Adam();
            initial_params = prior_sample(rng, post), maxiters = maxiters ÷ 2, g_tol = 1.0e-1
        )
        c20 = mapreduce(sum, +, chi2(post, xopt0)) / nd
        @info "Preliminary image $i/$(ntrials) done minimum χ²: $(c20)"

        xopt1, sol1 = comrade_opt(
            post, Adam();
            initial_params = xopt0, maxiters = maxiters ÷ 2, g_tol = 1.0e-1
        )
        c21 = mapreduce(sum, +, chi2(post, xopt1)) / nd
        @info "Best image $i/$(ntrials) done minimum χ²: $(c21)"
        return (sol0.objective < sol1.objective ? xopt0 : xopt1)
    end
    lmaps = logdensityof.(Ref(post), sols)
    valid = .!isnan.(lmaps)
    sols_v = sols[valid]
    lm_v = lmaps[valid]
    inds = sortperm(lm_v, rev = true)
    return sols_v[inds], lm_v[inds]
end


function plot_dterms(dLs, dRs, labels, layout; size=(1200, 800),
                     limits=((-0.3, 0.3), (-0.3, 0.3)),
                     markers=(keys(Makie.default_marker_map()) |> collect),
                     cL=Makie.wong_colors()[1], cR = Makie.wong_colors()[2],
                     markersize=10, alpha=0.5, plot_err=true
                     )
    ns = mapreduce(names, vcat, dLs) |> unique |> sort
    @assert length(ns) <= prod(layout)

    figure = Figure(;size=size,)
    axes = [Axis(figure[i,j],
             limits=limits,
             xticklabelfont="Computer Modern Roman",
             yticklabelfont="Computer Modern Roman",
             aspect = DataAspect(),
             ) for j in 1:layout[2], i in 1:layout[1]]


    axp = map(1:length(ns)) do i
            Pair(ns[i], axes[i])
    end |> Dict
    for n in ns
        text!(axp[n], 0.85, 0.95, align=(:right, :top), space=:relative, text=String(n), fontsize=20)
    end


    setproperty!.(axes[1, :], :ylabel, L"\mathrm{Im}(D_{L,R})")
    axes[1, end].xlabel = L"\mathrm{Re}(D_{L,R})"

    for i in eachindex(dLs,dRs)
        if labels[i] == "Truth"
            _plot_dterms!(axp, dLs[i], dRs[i], labels[i]; marker=markers[i], cL=:midnightblue, cR=:darkorange4, markersize, alpha, plot_err)
        else
            _plot_dterms!(axp, dLs[i], dRs[i], labels[i]; marker=markers[i], cL, cR, markersize, alpha, plot_err)
        end
    end

    # Now hide any empty plots
    map(axes) do ax
        if length(ax.scene.plots) == 0
            hidedecorations!(ax)
            hidespines!(ax)
        end
    end

    axislegend(axes[end],
                [PolyElement(color=(cL)), PolyElement(color=(cR))],
                [L"D_L", L"D_R"], framevisible=false, nbanks=2)
    mel = [MarkerElement(marker=markers[i], color=:black) for i in eachindex(dLs)]
    axislegend(axes[end], mel, labels, position=(0,0), nbanks=1, framevisible=false)
    rowgap!(figure.layout, 1)
    colgap!(figure.layout, 1)

    return figure, axes
end

function _plot_dterms!(axp, dL, dR, label; marker='o', markersize=10, alpha=1.0,
                       cL=Makie.wong_colors()[1], cR = Makie.wong_colors()[2],
                       plot_err = true)
    for n in names(dL)
        dLx, dLy = (real.(dL[!,n]), imag.(dL[!,n]))
        dRx, dRy = (real.(dR[!,n]), imag.(dR[!,n]))

        mLx = [mean(dLx)]
        mRx = [mean(dRx)]
        mLy = [mean(dLy)]
        mRy = [mean(dRy)]

        if plot_err
            sLx = [2*std(dLx)]
            sRx = [2*std(dRx)]
            sLy = [2*std(dLy)]
            sRy = [2*std(dRy)]
            errorbars!(axp[n], mLx, mLy, sLy, direction=:y, alpha=alpha, color=cL)
            errorbars!(axp[n], mLx, mLy, sLx, direction=:x, alpha=alpha, color=cL)
            errorbars!(axp[n], mRx, mRy, sRy, direction=:y, alpha=alpha, color=cR)
            errorbars!(axp[n], mRx, mRy, sRx, direction=:x, alpha=alpha, color=cR)
        end
        # CairoMakie.scatter!(axp[n], dLx, dLy, label=label, marker=marker, color=cL, alpha=alpha, markersize=markersize)
        # CairoMakie.scatter!(axp[n], dRx, dRy, label=label, marker=marker, color=cR, alpha=alpha, markersize=markersize)

        CairoMakie.scatter!(axp[n], mLx, mLy, label=label, marker=marker, color=cL, alpha=alpha, markersize=markersize)
        CairoMakie.scatter!(axp[n], mRx, mRy, label=label, marker=marker, color=cR, alpha=alpha, markersize=markersize)

    end
end

function dterm_table(chain)
    dLx = chain.instrument.dLre
    dRx = chain.instrument.dRre
    dLy = chain.instrument.dLim
    dRy = chain.instrument.dRim

    sites = Comrade.sites(dRy[1]) |> Tuple
    dL = NamedTuple{sites}.(Tuple.((map((x,y)->complex.(x,y), dLx, dLy))))
    dR = NamedTuple{sites}.(Tuple.((map((x,y)->complex.(x,y), dRx, dRy))))
    return DataFrame(dL), DataFrame(dR)
end