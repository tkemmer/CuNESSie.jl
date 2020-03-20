@inline _kcfg(n::Int) = _ -> (threads = 128, blocks = cld(n, 128))
