using VLBIFiles 

function build_data_uvfits(
        file::String, array::String;
        avg = "scan",
        ferr::Float64 = 0.005,
        trange = nothing,
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
    if !isnothing(trange)
        dcoh = filter(d->d.baseline.Ti ∈ trange, dcoh)
    end
    reset_mounts!(dcoh, array)

    dcoh = add_fractional_noise(dcoh, ferr)
    return dcoh
end



function build_data_dlist(
        file::String, array::String;
        avg = "scan",
        ferr::Float64 = 0.005,
        trange = nothing,
    )

    avg != "scan" && @warn "Only 'scan' averaging is supported for dlist files. Ignoring the --avg flag."

    dcoh0 = read_dlist(file, array)
    if !isnothing(trange)
        dcoh1 = filter(x -> trange[1] < x.baseline.Ti < trange[2], dcoh0)
    else
        dcoh1 = dcoh0
    end
    dcoh2 = add_fractional_noise(dcoh1, ferr)
    return dcoh2
end
