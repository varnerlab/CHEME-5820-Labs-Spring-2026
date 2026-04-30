
function eval_loss_accuracy(model, data_loader, device)
    loss = 0.
    acc = 0.
    ntot = 0
    for (g, y) in data_loader
        g, y = g |> device, y |> device
        n = size(y, 2)
        ŷ = model(g, g.ndata.x)
        loss += logitcrossentropy(ŷ, y) * n
        acc += sum(onecold(ŷ) .== onecold(y))
        ntot += n
    end
    return (loss = round(loss/ntot, digits=4), acc = round(acc*100/ntot, digits=2))
end

function train!(model; epochs=200, η=1e-2, infotime=10)

    # device = Flux.gpu # uncomment this for GPU training
	device = Flux.cpu
	model = model |> device
	opt_state = Flux.setup(Adam(η), model)


    function report(epoch)
        train = eval_loss_accuracy(model, train_loader, device)
        test = eval_loss_accuracy(model, test_loader, device)
        println("# epoch = $epoch")
        println("train = $train")
        println("test = $test")
    end

    report(0)
    for epoch in 1:epochs
        for (g, y) in train_loader
            g, y = g |> device, y |> device
            grads = Flux.gradient(model) do model
                ŷ = model(g, g.ndata.x)
                logitcrossentropy(ŷ, y)
            end
            Flux.Optimise.update!(opt_state, model, grads[1])
        end
		epoch % infotime == 0 && report(epoch)
    end
end


function (l::MyCustomConvolutionLayerModel)(g::GNNGraph, x::AbstractMatrix)
	message(xi, xj, e) = l.W2 * xj
	m = apply_edges(message, g, xj=x) # size [nout, num_edges]
	xnew = aggregate_neighbors(g, +, m) # size [nout, num_nodes]
	# m = propagate(message, g, +, xj=x) # equivalent to the 2 lines above
	return l.act.(l.W1*x .+ xnew .+ l.b)
end

Flux.@layer MyCustomConvolutionLayerModel

function build_model(; nin::Int, nh::Int, nout::Int, dropout_p::Float64)
    return GNNChain(
        input   = MyCustomConvolutionLayerModel(nin => nh, relu),
        hidden  = MyCustomConvolutionLayerModel(nh => nh, relu),
        output  = MyCustomConvolutionLayerModel(nh => nh),
        pool    = GlobalPool(mean),
        dropout = Dropout(dropout_p),
        dense   = Dense(nh, nout),
    )
end

function train_one_config(training_graphs, ytrain, testing_graphs, ytest;
                          nin::Int, nout::Int,
                          nh::Int, dropout_p::Float64,
                          batchsize::Int, η::Float64, epochs::Int)
    local_train_loader = DataLoader((training_graphs, ytrain),
                                    batchsize=batchsize, shuffle=true, collate=true)
    local_test_loader  = DataLoader((testing_graphs, ytest),
                                    batchsize=10, shuffle=false, collate=true)

    device = Flux.cpu
    model = build_model(nin=nin, nh=nh, nout=nout, dropout_p=dropout_p) |> device
    opt_state = Flux.setup(Adam(η), model)

    for _ in 1:epochs
        for (g, y) in local_train_loader
            g, y = g |> device, y |> device
            grads = Flux.gradient(model) do model
                ŷ = model(g, g.ndata.x)
                logitcrossentropy(ŷ, y)
            end
            Flux.Optimise.update!(opt_state, model, grads[1])
        end
    end

    train_eval = eval_loss_accuracy(model, local_train_loader, device)
    test_eval  = eval_loss_accuracy(model, local_test_loader,  device)
    return (train_acc = train_eval.acc, test_acc = test_eval.acc,
            train_loss = train_eval.loss, test_loss = test_eval.loss)
end
