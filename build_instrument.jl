function build_instrumentmodel_mixed(hier, file)
    if file
        if hier 
            @info "Using a hierarchical gain amplitude model dlist"
            return build_instrument_mixed_hier(;
                gain_amp_override = (;
                    LMT = IIDSitePrior(TrackSeg(), Normal(0.0, 1.0)),
                    SMA = IIDSitePrior(TrackSeg(), Normal(0.0, 0.5)),
                    NOEMA = IIDSitePrior(TrackSeg(), Normal(0.0, 0.5))
                ),
            )
        else
            @info "Using a standard mixed instrument model dlist"
            return build_instrument_mixed(;
                        gain_amp_override = (;
                            LMT = IIDSitePrior(IntegSeg(), Normal(0.0, 1.0)),
                            SMA = IIDSitePrior(IntegSeg(), Normal(0.0, 0.5)),
                            NOEMA = IIDSitePrior(IntegSeg(), Normal(0.0, 0.5))
                        ),
                )
        end

    else 
        if hier
            @info "Using a hierarchical gain amplitude model uvfits"
            return build_instrument_mixed_hier_uv(;
                gain_amp_override = (;
                    LM = IIDSitePrior(TrackSeg(), Normal(-1.0, 1.0)),
                    SM = IIDSitePrior(TrackSeg(), Normal(0.0, 0.5)),
                    NN = IIDSitePrior(TrackSeg(), Normal(0.0, 0.5))
                ),
            )
        else
            @info "Using a standard mixed instrument model uvfits"
            return build_instrument_mixed_uv(;
                        gain_amp_override = (;
                            LM = IIDSitePrior(IntegSeg(), Normal(0.0, 2.0)),
                            NN = IIDSitePrior(IntegSeg(), Normal(0.0, 0.5)),
                            GL = IIDSitePrior(IntegSeg(), Normal(0.0, 2.0)),
                            SM = IIDSitePrior(IntegSeg(), Normal(0.0, 0.5)),
                        ),
                    )
        end
    end
end

function build_instrumentmodel_pc(hier, file, frcal)
    if file
        frcal && @warn "dlist is never frcal-ed, ignorning the flag"
        if hier 
            @info "Using a hierarchical gain amplitude model dlist"
            return build_instrument_circular_hier(;
                gain_amp_override = (;
                    LMT = IIDSitePrior(TrackSeg(), Normal(0.0, 1.0)),
                    SMA = IIDSitePrior(TrackSeg(), Normal(0.0, 0.5)),
                    NOEMA = IIDSitePrior(TrackSeg(), Normal(0.0, 0.5))
                ),
                refsite = SingleReference(:ALMA, 0.0),
                frcal = false
            )
        else
            @info "Using a standard circular instrument model dlist"
            return build_instrument_circular_polconvert(;
                        gain_amp_override = (;
                            LMT = IIDSitePrior(IntegSeg(), Normal(0.0, 1.0)),
                            SMA = IIDSitePrior(IntegSeg(), Normal(0.0, 0.5)),
                            NOEMA = IIDSitePrior(IntegSeg(), Normal(0.0, 0.5))
                        ),
                refsite = SingleReference(:ALMA, 0.0),
                frcal = false
            )
        end

    else 
        if hier
            @info "Using a hierarchical gain amplitude model uvfits"
            return build_instrument_circular_hier(;
                gain_amp_override = (;
                    LM = IIDSitePrior(TrackSeg(), Normal(0.0, 1.0)),
                    SM = IIDSitePrior(TrackSeg(), Normal(0.0, 0.5)),
                    NN = IIDSitePrior(TrackSeg(), Normal(0.0, 0.5))
                ),
                frcal = frcal
            )
        else
            @info "Using a standard circular instrument model uvfits"
            return build_instrument_circular_noalmarot(;
                        gain_amp_override = (;
                            LM = IIDSitePrior(IntegSeg(), Normal(0.0, 2.0)),
                            NN = IIDSitePrior(IntegSeg(), Normal(0.0, 0.5)),
                            SM = IIDSitePrior(IntegSeg(), Normal(0.0, 0.5)),
                        ),
                        frcal = frcal
                    )
        end
    end
end