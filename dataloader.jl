using VLBIFiles 

function build_data_uvfits(
        file::String, array::String;
        avg = "scan",
        uvmin::Float64 = 0.0,
        ferr::Float64 = 0.005,
        trange = nothing,
        mixed = false
    )

    uvd = VLBIFiles.load(VLBIFiles.UVData, file)

    if avg == "scan"
        tavg = VLBI.GapBasedScans()
    else
        tavg = VLBI.FixedTimeIntervals(parse(Float64, avg)*VLBIFiles.Unitful.u"s")
    end


    # Unlike non-polarized tutorials, we also need an **array file** describing the antenna
    # feed rotation parameters. We pass it via the `arrayfile` keyword on the data product.
    dcoh = extract_table(
        uvd, Coherencies(;
            time_average = tavg,
        )
    )
    # Inflate noise by 1% and drop short (uvdist < 0.1 Gλ) baselines.
    dcoh= flag(d -> hypot(d.baseline.U, d.baseline.V) < uvmin, dcoh)
    if !isnothing(trange)
        dcoh = filter(d->d.baseline.Ti ∈ trange, dcoh)
    end

    reset_mounts!(dcoh, array)

    if mixed
        @info "Assuming ALMA is linear"
        dcoh = corr_polbasis(dcoh, :AA)
    end
    
    dcoh = add_fractional_noise(dcoh, ferr)
    return dcoh
end



function build_data_dlist(
        file::String, array::String;
        avg = "scan",
        uvmin::Float64 = 0.0,
        ferr::Float64 = 0.005,
        trange = nothing,
        mixed=true
    )

    avg != "scan" && @warn "Only 'scan' averaging is supported for dlist files. Ignoring the --avg flag."

    dcoh0 = read_dlist(file, array)
    dcoh1 = flag(x->uvdist(x) < uvmin, dcoh0)
    if !isnothing(trange)
        dcoh2 = filter(x -> trange[1] < x.baseline.Ti < trange[2], dcoh1)
    else
        dcoh2 = dcoh1
    end
    dcoh3 = add_fractional_noise(dcoh2, ferr)
    if mixed
        @info "Assuming ALMA is linear"
        dcoh3 = corr_polbasis(dcoh3, :AA)
    end
    return dcoh3
end
