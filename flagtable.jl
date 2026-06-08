using TOML

function read_flagtable(path::String)
    cfg = TOML.parsefile(path)
    sites_corr = Symbol.(get(cfg, "corr_polbasis", String[]))
    sites      = Symbol.(get(cfg, "sites",     String[]))
    baselines  = [Set(Symbol.(bl)) for bl in get(cfg, "baselines", Vector{String}[])]
    tranges    = [Tuple(Float64.(t)) for t in get(cfg, "tranges",  Vector{Float64}[])]
    uvranges   = [Tuple(Float64.(u)) for u in get(cfg, "uvranges", Vector{Float64}[])]

    for (i, bl) in enumerate(baselines)
        length(bl) == 2 || error("flag table: baselines[$i] must name two distinct sites, got $bl")
    end
    for (i, t)  in enumerate(tranges);  length(t)  == 2 || error("flag table: tranges[$i] must be [a, b], got $t");  end
    for (i, u)  in enumerate(uvranges); length(u)  == 2 || error("flag table: uvranges[$i] must be [a, b], got $u"); end

    @info "Flag table: corr_polbasis=$(length(sites_corr)) sites=$(length(sites)) baselines=$(length(baselines)) tranges=$(length(tranges)) uvranges=$(length(uvranges))"
    return (; corr_polbasis = sites_corr, sites, baselines, tranges, uvranges)
end

function apply_flagtable(dvis, cfg)
    for s in cfg.corr_polbasis
        @info "corr_polbasis: $s"
        dvis = corr_polbasis(dvis, s)
    end
    for s in cfg.sites
        n0 = length(dvis)
        dvis = flag(x -> s ∈ x.baseline.sites, dvis)
        @info "  flag site=$s dropped $(n0 - length(dvis)) datums"
    end
    for bl in cfg.baselines
        n0 = length(dvis)
        dvis = flag(x -> Set(x.baseline.sites) == bl, dvis)
        @info "  flag baseline=$(collect(bl)) dropped $(n0 - length(dvis)) datums"
    end
    for (a, b) in cfg.tranges
        n0 = length(dvis)
        dvis = flag(x -> a <= x.baseline.Ti <= b, dvis)
        @info "  flag trange=[$a, $b] dropped $(n0 - length(dvis)) datums"
    end
    for (a, b) in cfg.uvranges
        n0 = length(dvis)
        dvis = flag(x -> a <= uvdist(x) <= b, dvis)
        @info "  flag uvrange=[$a, $b] dropped $(n0 - length(dvis)) datums"
    end
    return dvis
end
