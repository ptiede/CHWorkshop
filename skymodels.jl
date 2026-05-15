using StatsFuns: logistic
abstract type PolRep end
abstract type PolModel <: PolRep end
struct Poincare <: PolModel end
struct PolExp <: PolModel end
struct TotalIntensity <: PolRep end
struct Matern end
struct MarkovRF{N} end
MarkovRF(n::Int) = MarkovRF{n}()

struct NonCenteredMRF{B}
    base::B
end

struct SRF{PS, P}
    ps::PS
    plan::P
end

# Hack because Enzyme seems to be choking when I have a `gamma` with the FFT
struct TBlobNN{T} <: VLBISkyModels.GeometricModel{T}
    slope::T
    function TBlobNN(slope::Number)
        T = typeof(slope)
        return new{T}(slope)
    end
end
VLBISkyModels.visanalytic(::Type{<:TBlobNN}) = VLBISkyModels.NotAnalytic()

VLBISkyModels.radialextent(m::TBlobNN) = 5 * m.slope / (m.slope - 2)

function VLBISkyModels.intensity_point(m::TBlobNN, p)
    x, y = VLBISkyModels._getxy(p)
    r² = x^2 + y^2
    VLBISkyModels.@unpack_params slope = m(p)
    return (1 + r² / slope)^(-(slope + 2) / 2)
end




struct ImagingModel{P, M, G, F, B, AG, C}
    mimg::M
    grid::G
    ftot::F
    base::B
    order::Int
end
Enzyme.EnzymeRules.inactive_type(::Type{<:ImagingModel}) = true

using StaticArrays
function fast_centroid(img::IntensityMap{<:Real, 2})
    x0 = zero(eltype(img))
    y0 = zero(eltype(img))
    dp = domainpoints(img)
    fs = Comrade._fastsum(img)
    @inbounds for i in CartesianIndices(img)
        x0 += dp[i].X * img[i]
        y0 += dp[i].Y * img[i]
    end
    return SVector(x0 / fs, y0 / fs)
end

fast_centroid(img::IntensityMap{<:StokesParams}) = fast_centroid(stokes(img, :I))


function ImagingModel(p::PolRep, mimg::M, grid, ftot; order=1, base=GMRF, center=centerfix(M), addgauss=false) where {M}
    b = prepare_base(base, grid, order)
    bt = typeof(b) === UnionAll ? Type{b} : typeof(b)
    return ImagingModel{typeof(p),M,typeof(grid),typeof(ftot),bt,addgauss,center}(mimg, grid, ftot, b, order)
end

prepare_base(b::Type{<:VLBIImagePriors.MarkovRandomField}, grid, order) = b
prepare_base(::NonCenteredMRF, grid, order) = (standardize(MarkovRandomFieldGraph(grid; order); flag = Comrade.VLBISkyModels.FFTW.EXHAUSTIVE))
prepare_base(::Matern, grid, order) = first(matern(size(grid)))

function ImagingModel(p::PolRep, mimg::IntensityMap, ftot; order = 1, base = GMRF, addgauss = false)
    return ImagingModel(p, mimg ./ sum(mimg), axisdims(mimg), ftot; order = order, base = base, addgauss = addgauss)
end

@inline prepare_base(ps::MarkovRF{N}, grid, order) where {N} = SRF(ps, StationaryRandomFieldPlan(grid))


@inline addgauss(::ImagingModel{P, M, G, F, B, AG}) where {P, M, G, F, B, AG} = AG
@inline center(::ImagingModel{P, M, G, F, B, AG, C}) where {P, M, G, F, B, AG, C} = C

getftot(m::ImagingModel{P, M, G, <:Real}, _) where {P, M, G} = m.ftot
getftot(::ImagingModel{P, M, G}, θ) where {P, M, G} = θ.ftot

function (m::ImagingModel{P})(θ, meta) where {P}
    mimg = make_mean(m.mimg, m.grid, θ)
    ftot = getftot(m, θ)
    if addgauss(m)
        fimg = ftot * (1 - θ.fg)
    else
        fimg = ftot
    end

    pmap = make_image(P, m.base, fimg, mimg, θ)
    if center(m)
        x0, y0 = fast_centroid(pmap)
        ms = shifted(ContinuousImage(pmap, DeltaPulse()), -x0, -y0)
    else
        ms = ContinuousImage(pmap, DeltaPulse())
    end

    model = addgauss(m, ftot, ms, θ)
    return model
end



@inline function addgauss(m::ImagingModel{<:PolModel}, ftot, ms, θ)
    if addgauss(m)
        (; fg, σg, τg, ξg, xg, yg, pg, pxg, pyg, pzg) = θ
        g = modify(Gaussian(), Stretch(σg, σg * (1 + τg)), Rotate(ξg / 2), Shift(xg, yg), Renormalize(ftot * fg))
        pr = sqrt(pxg^2 + pyg^2 + pzg^2) + 1.0e-6
        polg = PolarizedModel(g, (pg * pxg / pr) * g, (pg * pyg / pr) * g, (pg * pzg / pr) * g)
        return ms + polg
    else
        return ms
    end
end

@inline function addgauss(m::ImagingModel{<:TotalIntensity}, ftot, ms, θ)
    if addgauss(m)
        (; fg, σg, τg, ξg, xg, yg) = θ
        g = modify(Gaussian(), Stretch(σg, σg * (1 + τg)), Rotate(ξg / 2), Shift(xg, yg), Renormalize(ftot * fg))
        return ms + g
    else
        return ms
    end
end

function make_image(::Type{<:TotalIntensity}, t::VLBIImagePriors.NonCenteredMarkovTransform, ftot, mimg, θ)
    (; c, σ) = θ
    δ = centerdist(t, c.hyperparams, c.params)
    δ .*= σ
    img = IntensityMap(δ, axisdims(mimg))
    apply_fluctuations!(CenteredLR(), img, mimg, δ)
    bimg = baseimage(img)
    for i in eachindex(img)
        bimg[i] *= ftot
    end
    return img
end

@inline function make_image(::Type{<:Poincare}, ::Type{<:VLBIImagePriors.MarkovRandomField}, ftot, mimg, θ)
    (; c, σ, p, p0, pσ, angparams) = θ
    return make_poincare(ftot, mimg, σ .* c.params, p0, pσ, p.params, angparams)
end

@inline function make_image(::Type{<:PolExp}, ::Type{<:VLBIImagePriors.MarkovRandomField}, ftot, mimg, θ)
    (; a, b, c, d, σa, σb, σc, σd) = θ
    δa = similar(a.params)
    δb = similar(b.params)
    δc = similar(c.params)
    δd = similar(d.params)
    @inbounds for i in eachindex(δa, δb, δc, δd)
        δa[i] = σa * a.params[i]
        δb[i] = σb * b.params[i]
        δc[i] = σc * c.params[i]
        δd[i] = σd * d.params[i]
    end
    return make_pol2expimage(ftot, δa, δb, δc, δd, mimg)
end

@inline function make_image(::Type{<:PolExp}, t::VLBIImagePriors.NonCenteredMarkovTransform, ftot, mimg, θ)
    (; a, b, c, d, σa, σb, σc, σd) = θ
    δa = centerdist(t, a.hyperparams, a.params)
    δb = centerdist(t, b.hyperparams, b.params)
    δc = centerdist(t, c.hyperparams, c.params)
    δd = centerdist(t, d.hyperparams, d.params)

    for i in eachindex(δa, δb, δc, δd)
        δa[i] *= σa
        δb[i] *= σb
        δc[i] *= σc
        δd[i] *= σd
    end

    return make_pol2expimage(ftot, δa, δb, δc, δd, mimg)
end


@inline function make_image(::Type{<:Poincare}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (; c, σ, ρ, ν, p, p0, pσ, pν, pρ, angparams) = θ
    δ = trf(c, ρ, ν)
    pδ = trf(p, pρ, pν)
    for i in eachindex(δ)
        δ[i] *= σ
    end
    return make_poincare(ftot, mimg, δ, p0, pσ, pδ, angparams)
end

@inline function make_image(::Type{<:PolExp}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (; a, b, c, d, ρa, ρb, ρc, ρd, νa, νb, νc, νd, σa, σb, σc, σd) = θ
    δa = trf(a, ρa, νa)
    δb = trf(b, ρb, νb)
    δc = trf(c, ρc, νc)
    δd = trf(d, ρd, νd)
    @inbounds for i in eachindex(δa, δb, δc, δd)
        δa[i] *= σa
        δb[i] *= σb
        δc[i] *= σc
        δd[i] *= σd
    end
    return make_pol2expimage(ftot, δa, δb, δc, δd, mimg)
end

@inline function make_image(::Type{<:PolExp}, trf::SRF{<:MarkovRF{N}}, ftot, mimg, θ) where {N}
    (; a, b, c, d, ρa, ρb, ρc, ρd, σa, σb, σc, σd) = θ
    δa = genfield(StationaryRandomField(MarkovPS(ρa), trf.plan), a)
    δb = genfield(StationaryRandomField(MarkovPS(ρb), trf.plan), b)
    δc = genfield(StationaryRandomField(MarkovPS(ρc), trf.plan), c)
    δd = genfield(StationaryRandomField(MarkovPS(ρd), trf.plan), d)
    @inbounds for i in eachindex(δa, δb, δc, δd)
        δa[i] *= σa
        δb[i] *= σb
        δc[i] *= σc
        δd[i] *= σd
    end
    return make_pol2expimage(ftot, δa, δb, δc, δd, mimg)
end


@inline function make_image(::Type{<:TotalIntensity}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (; c, σ, ρ, ν) = θ
    δ = trf(c, ρ, ν)
    return make_stokesi(ftot, mimg, δ)
end

@inline function make_image(::Type{<:TotalIntensity}, trf::SRF{<:MarkovRF}, ftot, mimg, θ)
    (; c, σ, ρs) = θ
    ps = MarkovPS(ρs)
    δ = genfield(StationaryRandomField(ps, trf.plan), c)
    for i in eachindex(δ)
        δ[i] *= σ
    end
    return make_stokesi(ftot, mimg, δ)
end


@inline function make_image(::Type{<:TotalIntensity}, ::Type{<:VLBIImagePriors.MarkovRandomField}, ftot, mimg, θ)
    return make_stokesi(ftot, mimg, θ.σ .* θ.c.params)
end

@inline function make_stokesi(ftot, mimg, δ)
    stokesi = apply_fluctuations(CenteredLR(), mimg, δ)
    pstokesi = baseimage(stokesi)
    for i in eachindex(pstokesi)
        pstokesi[i] *= ftot
    end
    return stokesi
end


function make_poincare(ftot, mimg, δ, p0, pσ, pδ, angparams)
    stokesi = apply_fluctuations(CenteredLR(), mimg, δ)
    pstokesi = parent(stokesi)
    for i in eachindex(pstokesi)
        pstokesi[i] *= ftot
    end
    ptotim = logistic.(p0 .+ pσ .* pδ)
    pmap = PoincareSphere2Map(stokesi, ptotim, angparams)
    return pmap
end

function make_pol2expimage(ftot, a, b, c, d, mimg)
    # this allocated a whole new map so we can do things in place after
    δ = VLBISkyModels.PolExp2Map!(a, b, c, d, axisdims(mimg))
    brast = baseimage(δ)
    fr = zero(eltype(a))
    @inbounds for i in eachindex(mimg, brast)
        brast[i] *= mimg[i]
        fr += brast[i].I
    end

    for i in eachindex(brast)
        brast[i] *= ftot / fr
    end
    return δ
end


centerfix(::Type{<:Any}) = true

function skyprior(m::ImagingModel{P}; beamsize = μas2rad(20.0), overrides::Dict = Dict()) where {P}
    imgprior = genimgprior(P, m.base, m.grid, beamsize, m.order)
    mprior = genmeanprior(m.mimg)
    if addgauss(m)
        gprior = gengaussprior(P)
    else
        gprior = Dict()
    end

    if !(m.ftot isa Real)
        imgprior[:ftot] = m.ftot
    end

    prior = merge(imgprior, mprior, gprior)

    for k in keys(overrides)
        prior[k] = overrides[k]
    end


    return NamedTuple(prior)
end

function genimgprior(::Type{<:TotalIntensity}, base::VLBIImagePriors.NonCenteredMarkovTransform, grid, beamsize, order)
    cprior = VLBIImagePriors.StdNormal(size(grid))
    bs = beamsize / pixelsizes(grid).X
    dρ = truncated(InverseGamma(1.0, -log(0.01) * bs); lower = 1.0, upper = 2 * max(size(grid)...))

    default = Dict(
        :c => (hyperparams = dρ, params = cprior),
        :σ => truncated(Normal(0.0, 0.5); lower = 0.0),
    )
    return default
end

function genimgprior(::Type{<:TotalIntensity}, base::SRF{<:MarkovRF{N}}, grid, beamsize, order) where {N}
    bs = beamsize / step(grid.X)
    cprior = VLBIImagePriors.std_dist(base.plan)
    ρs = ntuple(Returns(Uniform(0.1, 1max(size(grid)...))), N)
    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 1.0); lower=0.0),
        :ρs => ρs,
    )
    return default
end



function genimgprior(::Type{<:Poincare}, base::Type{<:VLBIImagePriors.MarkovRandomField}, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base = base, order = order, lower = 4.0)
    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 0.5); lower = 0.0),
        :p => cprior,
        :p0 => Normal(-1.0, 2.0),
        :pσ => truncated(Normal(0.0, 0.5); lower = 0.0),
        :angparams => ImageSphericalUniform(size(cprior.priormap.cache)...)
    )
    return default
end

function genimgprior(::Type{<:PolExp}, base::Type{<:VLBIImagePriors.MarkovRandomField}, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base = base, order = order, lower = 4.0)
    default = Dict(
        :a => cprior,
        :b => cprior,
        :c => cprior,
        :d => cprior,
        :σa => truncated(Normal(0.0, 0.5); lower = 0.0),
        :σb => truncated(Normal(0.0, 0.5); lower = 0.0),
        :σc => truncated(Normal(0.0, 0.5); lower = 0.0),
        :σd => truncated(Normal(0.0, 0.05); lower = 0.0),
    )
    return default
end

function genimgprior(::Type{<:PolExp}, base::VLBIImagePriors.NonCenteredMarkovTransform, grid, beamsize, order)
    cprior = VLBIImagePriors.StdNormal(size(grid))
    bs = beamsize / pixelsizes(grid).X
    dρ = truncated(InverseGamma(1.0, -log(0.01) * bs); lower = 1.0, upper = 2 * max(size(grid)...))

    default = Dict(
        :a => (hyperparams = dρ, params = cprior),
        :b => (hyperparams = dρ, params = cprior),
        :c => (hyperparams = dρ, params = cprior),
        :d => (hyperparams = dρ, params = cprior),
        :σa => truncated(Normal(0.0, 0.5); lower = 0.0),
        :σb => truncated(Normal(0.0, 0.5); lower = 0.0),
        :σc => truncated(Normal(0.0, 0.5); lower = 0.0),
        :σd => truncated(Normal(0.0, 0.05); lower = 0.0),
    )
    return default
end


function genimgprior(::Type{<:Poincare}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize / step(grid.XL)
    cprior = VLBIImagePriors.std_dist(base)
    ρpr = truncated(InverseGamma(1.0, -log(0.1) * bs); lower = 4.0, upper = 2 * max(size(grid)...))
    νpr = truncated(InverseGamma(5.0, 9.0); lower = 0.1)

    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 0.5); lower = 0.0),
        :ρ => ρpr,
        :ν => νpr,
        :p => cprior,
        :ρp => ρpr,
        :νp => νpr,
        :p0 => Normal(-1.0, 2.0),
        :pσ => truncated(Normal(0.0, 0.5); lower = 0.0),
        :angparams => ImageSphericalUniform(size(cprior.priormap.cache)...)
    )
    return default
end

function genimgprior(::Type{<:PolExp}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize / step(grid.X)
    cprior = VLBIImagePriors.std_dist(base)
    ρpr = truncated(InverseGamma(1.0, -log(0.1) * bs); lower = 4.0, upper = 2 * max(size(grid)...))
    νpr = truncated(InverseGamma(5.0, 9.0); lower = 0.1)

    default = Dict(
        :a => cprior,
        :b => cprior,
        :c => cprior,
        :d => cprior,
        :σa => truncated(Normal(0.0, 0.5); lower = 0.0),
        :σb => truncated(Normal(0.0, 0.5); lower = 0.0),
        :σc => truncated(Normal(0.0, 0.5); lower = 0.0),
        :σd => truncated(Normal(0.0, 0.1); lower = 0.0),
        :ρa => ρpr,
        :νa => νpr,
        :ρb => ρpr,
        :νb => νpr,
        :ρc => ρpr,
        :νc => νpr,
        :ρd => ρpr,
        :νd => νpr
    )
    return default
end

function genimgprior(::Type{<:PolExp}, base::SRF{<:MarkovRF{N}}, grid, beamsize, order) where {N}
    cprior = VLBIImagePriors.std_dist(base.plan)
    ρs = ntuple(Returns(Uniform(0.1, max(size(grid)...))), N)
    default = Dict(
        :a => cprior,
        :b => cprior,
        :c => cprior,
        :d => cprior,
        :σa => truncated(Normal(0.0, 1.0); lower = 0.0),
        :σb => truncated(Normal(0.0, 0.5); lower = 0.0),
        :σc => truncated(Normal(0.0, 0.5); lower = 0.0),
        :σd => truncated(Normal(0.0, 0.1); lower = 0.0),
        :ρa => ρs,
        :ρb => ρs,
        :ρc => ρs,
        :ρd => ρs,

    )
    return default
end


function genimgprior(::Type{<:TotalIntensity}, base::Type{<:VLBIImagePriors.MarkovRandomField}, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base = base, order = order, lower = 4.0)
    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 0.5); lower = 0.0)
    )
    return default
end

function genimgprior(::Type{<:TotalIntensity}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize / step(grid.X)
    cprior = VLBIImagePriors.std_dist(base)
    ρpr = truncated(InverseGamma(1.0, -log(0.1) * bs); lower = 4.0, upper = 2 * max(size(grid)...))
    νpr = truncated(InverseGamma(5.0, 9.0); lower = 0.1)

    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 1.0); lower = 0.0),
        :ρ => ρpr,
        :ν => νpr
    )
    return default
end


function gengaussprior(::Type{<:PolModel})
    default = Dict(
        :fg => Uniform(0.0, 1.0),
        :σg => Uniform(μas2rad(250.0), μas2rad(1000.0)),
        :τg => Uniform(0.0, 7.0),
        :ξg => DiagonalVonMises(0.0, inv(1π^2)),
        :xg => Uniform(-μas2rad(10_000.0), μas2rad(10_000)),
        :yg => Uniform(-μas2rad(10_000.0), μas2rad(10_000.0)),
        :pg => Uniform(0.0, 1.0),
        :pxg => Normal(),
        :pyg => Normal(),
        :pzg => Normal()
    )
    return default
end

function gengaussprior(::Type{<:TotalIntensity})
    default = Dict(
        :fg => Uniform(0.0, 1.0),
        :σg => Uniform(μas2rad(250.0), μas2rad(1000.0)),
        :τg => Uniform(0.0, 7.0),
        :ξg => DiagonalVonMises(0.0, inv(1π^2)),
        :xg => Uniform(-μas2rad(10_000.0), μas2rad(10_000)),
        :yg => Uniform(-μas2rad(10_000.0), μas2rad(10_000.0)),
    )
    return default
end


function make_mean(mimg::IntensityMap, grid, θ)
    return mimg
end

function genmeanprior(::IntensityMap)
    return Dict()
end

struct MimgPlusBkg{M}
    mimg::M
    bkgd::M
    function MimgPlusBkg(mimg::IntensityMap)
        grid = axisdims(mimg)
        x0, y0 = phasecenter(grid)
        fovx, fovy = fieldofview(grid)
        pa = posang(grid)
        bkgd = intensitymap(modify(VLBISkyModels.GaussDisk(0.3), Stretch(fovx/2, fovy/2), Shift(-x0, -y0), Rotate(pa)), grid)
        return new{typeof(mimg)}(mimg./Comrade.flux(mimg), bkgd./Comrade.flux(bkgd))
    end
end

function make_mean(mimg::MimgPlusBkg, grid, θ)
    (; fb) = θ
    return mimg.mimg .* ((1 - fb)) .+ fb .* mimg.bkgd
end

function genmeanprior(::MimgPlusBkg)
    return Dict(:fb => Beta(1.0, 5.0))
end

struct GaussMean end
centerfix(::Type{<:GaussMean}) = true

const fwhmfac = 2*sqrt(2*log(2))
function make_mean(::GaussMean, grid, θ)
    (;fwhm) = θ
    m = modify(Gaussian(), Stretch(fwhm/fwhmfac))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    pmimg ./= sum(pmimg)
    return mimg
end

function genmeanprior(::GaussMean)
    return Dict(
            :fwhm => truncated(Normal(μas2rad(50.0), μas2rad(20.0)); lower=μas2rad(2.0), upper=μas2rad(100.0)),
        )
end



struct DblRingMean end
centerfix(::Type{<:DblRingMean}) = false

function make_mean(::DblRingMean, grid, θ)
    (; r0, ain, aout) = θ
    m = modify(RingTemplate(RadialDblPower(ain, aout), AzimuthalUniform()), Stretch(r0))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    pmimg .= pmimg ./ sum(pmimg)
    return mimg
end

function genmeanprior(::DblRingMean)
    return Dict(
        :r0 => Uniform(μas2rad(0.1), μas2rad(25.0)),
        :ain => Uniform(0.0, 10.0),
        :aout => Uniform(0.0, 10.0)
    )
end

struct DblRingWBkgd end
centerfix(::Type{<:DblRingWBkgd}) = false

function make_mean(::DblRingWBkgd, grid, θ)
    (; r0, ain, aout, fb) = θ
    m = modify(RingTemplate(RadialDblPower(ain, aout), AzimuthalUniform()), Stretch(r0))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    fbn = fb / (prod(size(grid)))
    pmimg .= pmimg ./ sum(pmimg) * ((1 - fb)) .+ fbn
    return mimg
end

function genmeanprior(::DblRingWBkgd)
    return Dict(
        :r0 => Uniform(μas2rad(10.0), μas2rad(25.0)),
        :ain => Exponential(3.0),
        :aout => Exponential(3.0) + 1,
        :fb => Beta(1.0, 5.0)
    )
end

struct TBlobMean end
centerfix(::Type{<:TBlobMean}) = true

function make_mean(::TBlobMean, grid, θ)
    (;fwhm, s) = θ
    m = modify(TBlobNN(s), Stretch(fwhm/fwhmfac))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    pmimg ./= sum(pmimg)
    return mimg
end

function genmeanprior(::TBlobMean)
    return Dict(
        :fwhm => truncated(Normal(μas2rad(50.0), μas2rad(20.0)); lower=μas2rad(10.0), upper=μas2rad(100.0)),
        :s => Uniform(1.0, 10.0)
        )
end

struct JetGauss{M}
    core::M
end
centerfix(::Type{<:JetGauss}) = true


function make_mean(mimg::JetGauss, grid, θ)
    (; r, τ, ξτ, x, y, fj) = θ
    img = intensitymap(modify(Gaussian(), Stretch(r, r * (1 + τ)), Rotate(ξτ / 2), Shift(x, y)), grid)
    fl = Comrade._fastsum(img)
    pimg = baseimage(img)
    pcore = baseimage(mimg.core)
    @inbounds for i in eachindex(pimg, pcore)
        pimg[i] = pcore[i] * (1 - fj) + pimg[i] / fl * fj
    end
    return img
end

function genmeanprior(m::JetGauss)
    fovx, fovy = fieldofview(m.core)
    x0, y0 = phasecenter(m.core)
    dx, dy = pixelsizes(m.core)
    return Dict(
        :r => Uniform(dx * 4, min(fovx, fovy) / 3),
        :τ => Uniform(0.0, 10.0),
        :ξτ => DiagonalVonMises(0.0, inv(π^2)),
        :x => Uniform(-fovx / 4 - x0, fovx / 4 - x0),
        :y => Uniform(-fovy / 4 - y0, fovy / 4 - y0),
        :fj => Beta(1.0, 5.0)
    )
end


struct GaussBkgdMean{M}
    bkgd::M
    function GaussBkgdMean(grid::RectiGrid)
        x0, y0 = phasecenter(grid)
        fovx, fovy = fieldofview(grid)
        pa = posang(grid)
        bkgd = intensitymap(modify(VLBISkyModels.GaussDisk(0.3), Stretch(fovx/2, fovy/2), Shift(-x0, -y0), Rotate(pa)), grid)
        return new{typeof(bkgd)}(bkgd./Comrade.flux(bkgd))
    end
end
centerfix(::Type{<:GaussBkgdMean}) = true

function make_mean(p::GaussBkgdMean, grid, θ)
    (;fwhm, fb) = θ
    m = modify(Gaussian(), Stretch(fwhm/fwhmfac))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    pf = Comrade._fastsum(pmimg)
    @inbounds for i in eachindex(pmimg)
        pmimg[i] = pmimg[i] * (1-fb) / pf + p.bkgd[i] * fb
    end
    return mimg
end

function genmeanprior(::GaussBkgdMean)
    return Dict(
        :fwhm => truncated(Normal(μas2rad(50.0), μas2rad(20.0)); lower=μas2rad(20.0), upper=μas2rad(100.0)),
        :fb => Uniform(0.0, 1.0)
    )
end


