import numpy as np
import abc

class Base(object):
    def gradient_descent(self, params=np.array([2.0, 2.0]),numIter=2001,
        verbose=False, set_zero=60,
        step_lengths=[1e-8, 1e-7, 1e-5, 1e-3, 1e-2, .1, 1,2]
        ):
        '''
        Performs gradient descent to fit the parameters of our distribution.
        args:
            numIter: The number of iterations gradient descent should run for.
            params: The starting parameters where it starts.
            verbose: To print progress in iterations or not.            
            step_lengths: The step lengths along the gradient the algorithm should check 
                          and make the step with the best improvement in objective function.
        '''
        self.step_lens = {}
        for i in range(numIter):
            directn = self.grad(params)
            if max(abs(directn)) < 1e-3:
                self.final_loglik = self.loglik(params)
                return params
            # In 20% of the iterations, we set all but one of the gradient
            # dimensions to zero.
            # This works better in practice.
            if i % 100 > set_zero:
                # Randomly set one coordinate to zero.
                directn[np.random.choice(len(params), 1)[0]] = 0
            params2 = params + 1e-10 * directn
            lik = self.loglik(params2)
            alp_used = step_lengths[0]
            for alp1 in step_lengths:
                params1 = params + alp1 * directn
                if(min(params1) > 0):
                    lik1 = self.loglik(params1)
                    if(lik1 > lik and np.isfinite(lik1)):
                        lik = lik1
                        params2 = params1
                        alp_used = alp1
            if alp_used in self.step_lens:
                self.step_lens[alp_used] += 1
            else:
                self.step_lens[alp_used] = 1
            params = params2
            if i % 100 == 0 and verbose:
                    print("Itrn " + str(i) + " ,obj fn: " 
                        + str(lik) + " \nparams = " + 
                        str(params) + " \ngradient = " + str(directn) + 
                            "\nstep_len=" + str(alp_used))
                    print("\n########\n")
        return params

