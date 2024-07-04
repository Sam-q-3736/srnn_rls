import numpy as np

class BaseModel:
    def create_default_params():
        raise NotImplementedError

    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError



    # def plot_spk_rasts(spk_rast, inds):
    #     spk_inds, spk_t = np.nonzero(spk_rast)
    #     spk_times = []
    #     for idx in np.unique(spk_inds):
    #         spk_times.append(spk_t[spk_inds == idx])
    #     plt.eventplot(spk_times[inds]);


class RateModel(BaseModel):
    # Target generating network:
    # tau * dx^D/dt = -x^D + J^D * H(x^d) + u_fout * f_out(t) + u_in * f_in(t) + u_hint * f_hint(t)
    # Task performing network:
    # tau * dx/dt = -x + J * H(x) + u_in * f_in(t)
    # Task output:
    # z(t) = w^T * H(x(t))
    # Optimizing J:
    # Cost Function:
    # C_J = <|J * H(x(t)) - J^D * H(x^D(t)) - u_out * f_out(t)|^2>
    # Error:
    # e(t) = J(t-dt) * H(x(t)) - J^D * H(x^D(t)) - u_out * f_out(t)
    # new J(t) = J(t-dt) - e(t)^T * P(t) * H(x(t))
    # P(t) = P(t-dt) - ( P(t-dt) * H(x(t)) * H(x(t))^T * P(t-dt) ) / ( 1 + H(x(t))^T * P(t-dt) * H(x(t)) )
    # P(0) = I / lambda
    # Optimizing w:
    # Cost Function:
    # C_w = < ( z(t) - f_out(t) )^2 >
    # C_w = < ( w^T * H(x(t)) - f_out(t) )^2 >
    # Error:
    # e(t) = w^T(t-dt) * H(x(t)) - f_out(t)
    # new w(t) = w(t-dt) - e(t) * P(t) * H(x(t))
    # P(t) = P(t-dt) - ( P(t-dt) * H(x(t)) * H(x(t))^T * P(d-dt) ) / ( 1 + H(x(t))^T * P(t-dt) * H(x(t)) )
    # P(0) = I / lambda
    
    def create_default_params():
        p = {
            'N': 300,       # Number of neurons
            'dt': 0.001,    # in seconds
            'tau': 0.01,    # in seconds
            'lambda': 1,    # learning rate (between 1 - 100 are supposed to be good values)
            'n_in': 1,      # number of inputs
            'n_out': 1,     # number of outputs
            'n_hint': 0,    # number of hints
            'update_every': 2,  # average number of steps between updates
        }
        return p

    def __init__(self, params, inp_out_hint_fn, **kwargs):
        self.p = params
        self.x = np.random.randn(self.p['N'])
        self.model = dict(
            J = np.zeros((self.p['N'], self.p['N'])),

        )
        self.inp_out_hint_fn = inp_out_hint_fn
    
    def run(self):
        inp, out, hint = self.inp_out_hint_fn(p['dt'])
        def dx(dt, x, total_inputs):
            # return dt/self.p['tau'] - x + self.J * np.tanh(x) + 
            pass

    def train(self):
        pass



if __name__ == "__main__":
    p = RateModel.create_default_params()
    mdl = RateModel(p)
    print(mdl.p)