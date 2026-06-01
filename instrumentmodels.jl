@inline function gain(x)
    gR = exp(x.lgR + 1im * x.gpR)
    lgrat = x.lgratμ + x.lgratσ * x.lgrat
    gprat = x.gprat + x.gpratμ
    gL = gR * exp(lgrat + 1im * gprat)
    return gR, gL
end

@inline function gainc(x)
    gR = exp(x.lgR + 1im * x.gpR)
    lgrat = x.lgratμ + x.lgratσ * x.lgrat
    gprat = x.gprat
    gL = gR * exp(lgrat + 1im * gprat)
    return gR, gL
end

@inline fgain(x) = exp(x.lg + 1im * x.gp)
@inline fgainhier(x) = exp(x.lgμ + x.lgσ * x.lg + 1im * x.gp)

function build_instrument(; gainamp_sigma=0.2, gain_amp_override=(; LM=IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))), refant=SEFDReference(0.0))
    G = SingleStokesGain(fgain)

    intprior = (
        lg=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, gainamp_sigma)); gain_amp_override...),
        gp=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))), refant=refant, phase=true),
    )

    return InstrumentModel(G, intprior)
end


@inline function gainh(x)
    lgR = x.lgRμ + x.lgRσ * x.lgR
    gR = exp(lgR + 1im * x.gpR)
    lgrat = x.lgratμ + x.lgratσ * x.lgrat
    gprat = x.gprat + x.gpratμ
    gL = gR * exp(lgrat + 1im * gprat)
    return gR, gL
end

@inline function gainrat(x)
    gR = exp(x.lgR + 1im * x.gpR)
    lgrat = x.lgratμ + x.lgratσ * x.lgrat
    gprat = x.gpratμ + x.gprat
    gL = gR * exp(lgrat + 1im * gprat)
    return gR, gL
end


@inline function gainsimple(x)
    gR = exp(x.lgR + 1im * x.gpR)
    lgrat = x.lgratμ + x.lgratσ * x.lgrat
    gL = gR * exp(lgrat + 1im * x.gprat)
    return gR, gL
end


@inline function gainfo(x)
    gR = exp(x.lgσ * x.lgz + 1im * x.gp)
    return gR
end


@inline jfr(g, d, r) = adjoint(r) * g * d * r
@inline jnofr(g, d, r) = g * d * r

@inline function dterm(x)
    dR = complex(x.dRre, x.dRim)
    dL = complex(x.dLre, x.dLim)
    return dR, dL
end

@inline function dtermh(x)
    dR = complex(x.dRre + x.μdRre, x.dRim + x.μdRim)
    dL = complex(x.dLre + x.μdLre, x.dLim + x.μdLim)
    return dR, dL
end

@inline function dtermth(x)
    dR = complex(x.dRre * x.σdRre + x.μdRre, x.σdRim * x.dRim + x.μdRim)
    dL = complex(x.dLre * x.σdLre + x.μdLre, x.σdLim * x.dLim + x.μdLim)
    return dR, dL
end

# function rotmat(ϕ)
#     c = cos(ϕ)
#     s = sin(ϕ)
#     return @SMatrix [c -s; s c]
# end

# @inline function vlbadterm(x)
#     R1 = rotmap(x.ϕ)
#     po = exp(1im*(x.η + π/4))
#     Q = @SMatrix [po 0; 0 conj(po)]
# end


@inline function stokesgain(x)
    return exp(x.lg + 1im * x.gp)
end

function build_instrument(; gainamp_sigma=0.15, refsite=SEFDReference(0.0), gain_amp_override=(;))
    G = SingleStokesGain(stokesgain)

    intprior = (
        lg=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, gainamp_sigma)); gain_amp_override...),
        gp=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=refsite),
    )

    return InstrumentModel(G, intprior)
end

function build_instrument_circular_fo(;
    gainamp_sigma=0.1, frcal=true, gain_amp_override=(;),
    AAvard=false
)

    G = SingleStokesGain(gainfo)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end

    if !AAvard
        AAseg = TrackSeg()
    else
        AAseg = IntegSeg()
    end

    intprior = (
        lgz=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); gain_amp_override...),
        lgσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(gainamp_sigma))),
        gp=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
    )

    return InstrumentModel(J, intprior)
end

function build_instrument_circular(;
    refsite=SingleReference(:AA, 0.0), frcal=true, gain_amp_override=(;),
    AAvard=false
)

    G = JonesG(gainc)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end

    if !AAvard
        AAseg = TrackSeg()
    else
        AAseg = IntegSeg()
    end

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.4)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=refsite, phase=true),
        gprat=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(1.0^2)));
            PT=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.05^2))), refant=refsite, phase=true),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
    )

    return InstrumentModel(J, intprior)
end

function build_instrument_circular_vard(;
    refsite=SingleReference(:AA, 0.0), frcal=true, gain_amp_override=(;),
    AAvard=false
)

    G = JonesG(gainc)

    D = JonesD(dtermh)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end

    if !AAvard
        AAseg = TrackSeg()
    else
        AAseg = IntegSeg()
    end

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.4)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); refant=refsite),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=refsite, phase=true),
        gprat=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(1.0^2)));
            PT=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(deg2rad(5.0)^2))), phase=true),
        dRre=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.01))),
        dRim=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.01))),
        dLre=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.01))),
        dLim=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.01))),
        μdRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.3))),
        μdRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.3))),
        μdLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.3))),
        μdLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.3))),
    )

    return InstrumentModel(J, intprior)
end


function build_instrument_circular_hiervard(;
    refsite=SingleReference(:AA, 0.0), frcal=true, gain_amp_override=(;),
    AAvard=false
)

    G = JonesG(gainc)

    D = JonesD(dtermh)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end

    if !AAvard
        AAseg = TrackSeg()
    else
        AAseg = IntegSeg()
    end

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.4)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); refant=SingleReference(:PT, 0.0)),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=refsite, phase=true),
        gprat=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(1.0^2)));
            PT=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(deg2rad(5.0)^2))), refant=refsite, phase=true),
        dRre=ArrayPrior(IIDSitePrior(IntegSeg(), Normal()); refant=SingleReference(:PT, 0.0)),
        dRim=ArrayPrior(IIDSitePrior(IntegSeg(), Normal()); refant=SingleReference(:PT, 0.0)),
        dLre=ArrayPrior(IIDSitePrior(IntegSeg(), Normal()); refant=SingleReference(:PT, 0.0)),
        dLim=ArrayPrior(IIDSitePrior(IntegSeg(), Normal()); refant=SingleReference(:PT, 0.0)),
        σdRre=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.005))),
        σdRim=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.005))),
        σdLre=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.005))),
        σdLim=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.005))),
        μdRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.3)),),
        μdRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.3)),),
        μdLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.3)),),
        μdLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.3)),),
    )

    return InstrumentModel(J, intprior)
end



function build_instrument_circular_noalmarot(;
    refsite=SingleReference(:AA, π / 2), frcal=false,
    gain_amp_override=(; LM=IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
    AAvard=false
)

    G = JonesG(gainc)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end

    if !AAvard
        AAseg = TrackSeg()
    else
        AAseg = IntegSeg()
    end

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.2)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))); refant=refsite, phase=true),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15)), AA=IIDSitePrior(AAseg, Normal(0.0, 0.2))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15)), AA=IIDSitePrior(AAseg, Normal(0.0, 0.2))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15)), AA=IIDSitePrior(AAseg, Normal(0.0, 0.2))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15)), AA=IIDSitePrior(AAseg, Normal(0.0, 0.2))),
    )

    return InstrumentModel(J, intprior)
end


function build_instrument_circular_vlba(;
    refsite=SingleReference(:AA, 0.0), frcal=true, gain_amp_override=(;),
    AAvard=false
)

    G = JonesG(gainc)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end


    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.4)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))); PT=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.05^2))), phase=true),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
    )

    return InstrumentModel(J, intprior)
end


function build_instrument_circular_polconvert(;
    refsite=SingleReference(:AA, 0.0), frcal=true, gain_amp_override=(; LM=IIDSitePrior(TrackSeg(), Normal(0.0, 1.0))),
    AAvard=false
)

    G = JonesG(gainc)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end

    if !AAvard
        AAseg = TrackSeg()
    else
        AAseg = IntegSeg()
    end

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.4)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))); refant=refsite, phase=true),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15)), AA=IIDSitePrior(IntegSeg(), Normal(0.0, 0.2))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15)), AA=IIDSitePrior(IntegSeg(), Normal(0.0, 0.2))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15)), AA=IIDSitePrior(IntegSeg(), Normal(0.0, 0.2))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15)), AA=IIDSitePrior(IntegSeg(), Normal(0.0, 0.2))),
    )

    return InstrumentModel(J, intprior)
end


function build_instrument_circular_hier(;
    refsite=SingleReference(:AA, 0.0), gainamp_sigma=0.2, frcal=true, gain_amp_override=(; LM=IIDSitePrior(TrackSeg(), Normal(0.0, 1.0))),
    AAvard=false
)

    G = JonesG(gainh)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end

    if !AAvard
        AAseg = TrackSeg()
    else
        AAseg = IntegSeg()
    end

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), TDist(4))),
        lgRμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); gain_amp_override...),
        lgRσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(gainamp_sigma))),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.1^2))); refant=refsite, phase=true),
        gpratμ=ArrayPrior(IIDSitePrior(TrackSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=refsite),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
    )

    return InstrumentModel(J, intprior)
end


function build_instrument_circsimple(;
    gainamp_sigma=0.2, refsite=SingleReference(:AA, 0.0), frcal=true, gain_amp_override=(; LM=IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
    AAvard=false
)

    G = JonesG(gainsimple)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end

    if !AAvard
        AAseg = TrackSeg()
    else
        AAseg = IntegSeg()
    end

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, gainamp_sigma)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))); refant=refsite, phase=false),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.15))),
    )

    return InstrumentModel(J, intprior)
end


function build_instrument_mixed(; gainamp_sigma=0.4, frcal=false, gain_amp_override=(;))

    G = JonesG(gain)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    J = JonesSandwich(jnofr, G, D, R)

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, gainamp_sigma)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.1^2))); SMA=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))), phase=true),
        gpratμ=ArrayPrior(IIDSitePrior(TrackSeg(), DiagonalVonMises(0.0, inv(π^2)))),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); 
                        ALMA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01)),
                        HAY = IIDSitePrior(TrackSeg(), Normal(0.0, 0.05))
                        ),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); 
                        ALMA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01)),
                        HAY = IIDSitePrior(TrackSeg(), Normal(0.0, 0.05))
                        ),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); 
                        ALMA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01)),
                        HAY = IIDSitePrior(TrackSeg(), Normal(0.0, 0.05))
                        ),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); 
                        ALMA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01)),
                        HAY = IIDSitePrior(TrackSeg(), Normal(0.0, 0.05))
                        ),
    )

    return InstrumentModel(J, intprior)
end

function build_instrument_mixed_uv(; gainamp_sigma=0.2, frcal=false, gain_amp_override=(;))

    G = JonesG(gain)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    J = JonesSandwich(jnofr, G, D, R)

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, gainamp_sigma)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(
            IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); 
            refant=SEFDReference(0.0), 
            phase=true
        ),
        gprat=ArrayPrior(
            IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.1^2))); 
            SW=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))), phase=true
        ),
        gpratμ=ArrayPrior(IIDSitePrior(TrackSeg(), DiagonalVonMises(0.0, inv(π^2)))),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
    )

    return InstrumentModel(J, intprior)
end

function build_instrument_hopspc_uv(; gainamp_sigma=0.2, frcal=false, gain_amp_override=(;))

    G = JonesG(gain)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    J = JonesSandwich(jfr, G, D, R)

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, gainamp_sigma)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); refant = SingleReference(:AA, 0.0)),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat=ArrayPrior(
                    IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.1^2))); 
                    SW=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))), 
                    refant=SingleReference(:AA, 0.0), phase=true
                ),
        gpratμ=ArrayPrior(
                IIDSitePrior(TrackSeg(), DiagonalVonMises(0.0, inv(π^2))); 
                refant=SingleReference(:AA, 0.0)
                ),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
    )

    return InstrumentModel(J, intprior)
end

function build_instrument_hopspc_uv2(; gainamp_sigma=0.2, frcal=false, gain_amp_override=(;), gref = SingleReference(:AA, 0.0))

    G = JonesG(gain)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    J = JonesSandwich(jfr, G, D, R)

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, gainamp_sigma)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); refant = gref),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat=ArrayPrior(
                    IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.1^2))); 
                    SW=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))), 
                    refant=SingleReference(:AA, 0.0), phase=true
                ),
        gpratμ=ArrayPrior(
                IIDSitePrior(TrackSeg(), DiagonalVonMises(0.0, inv(π^2))); 
                refant=SingleReference(:AA, π/2)
                ),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
    )

    return InstrumentModel(J, intprior)
end



function build_instrument_mixed_varaa(; gainamp_sigma=0.4, refsite=SingleReference(:AA, 0.0), frcal=false, gain_amp_override=(;))

    G = JonesG(gain)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, gainamp_sigma)); gain_amp_override...),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.1^2))); SMA=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))), phase=true),
        gpratμ=ArrayPrior(IIDSitePrior(TrackSeg(), DiagonalVonMises(0.0, inv(π^2)))),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(IntegSeg(), Normal(0.0, 0.1))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(IntegSeg(), Normal(0.0, 0.1))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(IntegSeg(), Normal(0.0, 0.1))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(IntegSeg(), Normal(0.0, 0.1))),
    )

    return InstrumentModel(J, intprior)
end

function build_instrument_hopspc_varaa(; gainamp_sigma=0.4, refsite=SingleReference(:ALMA, 0.0), frcal=false, gain_amp_override=(;))

    G = JonesG(gainh)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end

    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), TDist(4))),
        lgRμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.3)); gain_amp_override...),
        lgRσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat=ArrayPrior(
            IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.1^2)));
            SMA=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))),
            phase=true
        ),
        gpratμ=ArrayPrior(IIDSitePrior(TrackSeg(), DiagonalVonMises(0.0, inv(π^2)))),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(IntegSeg(), Normal(0.0, 2.0))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(IntegSeg(), Normal(0.0, 2.0))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(IntegSeg(), Normal(0.0, 2.0))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(IntegSeg(), Normal(0.0, 2.0))),
    )

    return InstrumentModel(J, intprior)
end

function build_instrument_mixed_hier_uv(; gainamp_sigma=0.2, gain_amp_override=(;))

    G = JonesG(gainh)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    J = JonesSandwich(jnofr, G, D, R)


    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), TDist(4))),
        lgRμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, gainamp_sigma)); gain_amp_override...),
        lgRσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.5))),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(
            IIDSitePrior(TrackSeg(), Normal(0.0, 0.2));
        ),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(
            IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2)));
            refant=SEFDReference(0.0), phase=true
        ),
        gprat=ArrayPrior(
            IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.1^2)));
            SM=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))), phase=true
        ),
        gpratμ=ArrayPrior(IIDSitePrior(TrackSeg(), DiagonalVonMises(0.0, inv(π^2)))),
        dRre=ArrayPrior(
            IIDSitePrior(TrackSeg(), Normal(0.0, 0.2));
            AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))
        ),
        dRim=ArrayPrior(
            IIDSitePrior(TrackSeg(), Normal(0.0, 0.2));
            AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))
        ),
        dLre=ArrayPrior(
            IIDSitePrior(TrackSeg(), Normal(0.0, 0.2));
            AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))
        ),
        dLim=ArrayPrior(
            IIDSitePrior(TrackSeg(), Normal(0.0, 0.2));
            AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))
        ),
    )

    return InstrumentModel(J, intprior)
end


function gainm(x)
    gR = exp(x.lgRμ + x.lgRσ .* x.lgR) #-ve b/c we truncate lgR to be +ve
    lgrat = x.lgratμ + x.lgratσ * x.lgrat
    gprat = x.gprat + x.gpratμ
    gL = gR * exp(lgrat + 1im * gprat)
    return gR, gL
end

function build_instrument_mixed_mhier_uv(; gainamp_sigma=0.2, gain_amp_override=(;))

    G = JonesG(gainh)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    J = JonesSandwich(jnofr, G, D, R)


    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), truncated(Normal(), lower=0.0))),
        lgRμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, gainamp_sigma)); gain_amp_override...),
        lgRσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(
            IIDSitePrior(TrackSeg(), Normal(0.0, 0.2));
            refant=SingleReference(:ALMA, 0.0)
        ),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(
            IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2)));
            refant=SEFDReference(0.0), phase=true
        ),
        gprat=ArrayPrior(
            IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.1^2)));
            SM=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))), phase=true
        ),
        gpratμ=ArrayPrior(IIDSitePrior(TrackSeg(), DiagonalVonMises(0.0, inv(π^2)))),
        dRre=ArrayPrior(
            IIDSitePrior(TrackSeg(), Normal(0.0, 0.2));
            AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))
        ),
        dRim=ArrayPrior(
            IIDSitePrior(TrackSeg(), Normal(0.0, 0.2));
            AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))
        ),
        dLre=ArrayPrior(
            IIDSitePrior(TrackSeg(), Normal(0.0, 0.2));
            AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))
        ),
        dLim=ArrayPrior(
            IIDSitePrior(TrackSeg(), Normal(0.0, 0.2));
            AA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))
        ),
    )

    return InstrumentModel(J, intprior)
end



function build_instrument_mixed_hier(; gainamp_sigma=0.4, frcal=false, gain_amp_override=(;))

    G = JonesG(gainh)

    D = JonesD(dterm)

    R = JonesR(; add_fr=true)

    if frcal
        J = JonesSandwich(jfr, G, D, R)
    else
        J = JonesSandwich(jnofr, G, D, R)
    end


    intprior = (
        lgR=ArrayPrior(IIDSitePrior(IntegSeg(), TDist(4))),
        lgRμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, gainamp_sigma)); gain_amp_override...),
        lgRσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        lgrat=ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
        lgratμ=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
        lgratσ=ArrayPrior(IIDSitePrior(TrackSeg(), Exponential(0.2))),
        gpR=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true),
        gprat=ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.1^2))); SMA=IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(0.5^2))), phase=true),
        gpratμ=ArrayPrior(IIDSitePrior(TrackSeg(), DiagonalVonMises(0.0, inv(π^2)))),
        dRre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
        dRim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
        dLre=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
        dLim=ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2)); ALMA=IIDSitePrior(TrackSeg(), Normal(0.0, 0.01))),
    )

    return InstrumentModel(J, intprior)
end
