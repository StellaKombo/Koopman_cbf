function err_bnd = koopman_error_bound_hankel_mu(mu, L, e_max, tt, K_pows, C, bound_type)
    global Ts;
    p = 2;    
    CA_norm = 0;
    err_bnd = cell(1, length(tt)-1);
    
    for k = 1:length(tt)-1
        CA_norm = CA_norm + norm(C*K_pows{k}, p);
        switch bound_type
            case 0
                % Simple cumulative bound
                err_bnd{k} = norm(e_max, p) * CA_norm;
            case 1
                % Add Lipschitz + sampling density term
                err_bnd{k} = norm(C*K_pows{k}, p) * L * mu + norm(e_max, p) * CA_norm + mu;
        end
    end 
end
