classdef multNIM
	% Class implementation of separable NIM based on NIM class. 
	
	properties
		nim;		% NIM or sNIM struct that contains the normal subunits
		Msubunits;  % actual NIM/sNIM subunits that should multiply Mtargets
		Mtargets;   % targets of Msubunits in the NIM
	end	
	
	properties (Hidden)
		normal_subs;
		%allowed_reg_types = {'nld2','d2xt','d2x','d2t','l2','l1'}; %set of allowed regularization types		
		%allowed_spkNLs = {'lin','rectpow','exp','softplus','logistic'}; %set of NL functions currently implemented		
		%allowed_noise_dists = {'poisson','bernoulli','gaussian'}; %allowed noise distributions
		version = '0.2';    %source code version used to generate the model
		create_on = date;    %date model was generated
		%min_pred_rate = 1e-50; %minimum predicted rate (for non-negative data) to avoid NAN LL values
		%opt_check_FO = 1e-3; %threshold on first-order optimality for fit-checking
	end
		
	%% METHODS DEFINED IN SEPARATE FILES
  methods 
		%[] = display_model(nim,Robs,Xstims,varargin); %display current model
	end
	
	methods (Static)
    %Xmat = create_time_embedding( stim, params ) %make time-embedded stimulus
	end
	methods (Static, Hidden)
		%Tmat = create_Tikhonov_matrix( stim_params, reg_type ); %make regularization matrices
	end
	
	
methods
        
	%% CONSTRUCTOR
	function mnim = multNIM( nim, Msubunits, Mtargets )
	% Usage: mnim = multNIM( nim, <Msubunits, Mtargets> )
	%
	% multNIM must be initialized with a regular NIM. Additional optional arguments specify which subunits
	% "Msubunits" should be made to be multiplicative, and what their targets are (i.e. what they are multiplying)
	
		mnim.nim = nim;
		mnim.Msubunits = [];
		mnim.Mtargets = [];
		%mnim.fit_props = [];
		Nsubs = length(nim.subunits);
		%mnim.normal_subs = 1:Nsubs;
		if nargin < 2
			return
		end
		assert( nargin == 3, 'Must specify targets as well as subunits' )
		%assert( sum( Msubunits <= Nsubs), 'Msubunits out of range.' )
		assert( length(Msubunits) == length(Mtargets), 'Mtargets out of range.' )
		%assert( isempty(Msubunits,Mtargets), 'Mtargets cannot be multiplicative subunits' )
		mnim.Msubunits = Msubunits;
		%mnim.normal_subs = setdiff(Nsubs, Msubunits );
		mnim.Mtargets = Mtargets;
	end
	
	function mnim = add_Msubunit( mnim, subunit, Mtarget )
	% Usage: mnim = mnim.add_Msubunit( subunit, Mtarget )
	%
	% INPUTS:
	%		subunit: this can either be a subunit class to add to the NIM model, or a number of an existing
	%							subunit to clone for a multiplicative subunit 
	%		Mtarget: subunit that new Msubunit multiplies

		%assert( ismember(mnim.normal_subs, Mtarget), 'Mtarget must be an existing subunit.' )
		assert( Mtarget <= length(mnim.nim.subunits), 'Mtarget must be an existing subunit.' )
		if strcmp( class( subunit ), 'double' ) 
			%assert( ismember(mnim.normal_subs,subunit), 'subunit to be multiplicative must be a normal subunit' )
			assert( subunit <= length(mnim.nim.subunits), 'subunit to be multiplicative must be a normal subunit' )
			Msubunit = mnim.subunits(subunit);
			% Default nonlinearity/scaling?
			%mnim.normal_subs = setdiff( mnim.normal_subs, Msubunit );
		else
			mnim.Msubunits = cat(1, mnim.Msubunits, subunit );
		end
	
		mnim.Mtargets(end+1) = Mtarget;
	end
	
	%%
	function [sub_outs,fgadd,fgmult] = subunit_outputs( mnim, stims )
	% Usage: [sub_outs,fgadd,fgmult] = subunit_outputs( mnim, stims )
	% 
	% Calculate output of all subunits (addxmult) and separate additive and multiplicative
	
		[~,~,mod_internals] = mnim.nim.eval_model( zeros(size(stims{1},1),1), stims );
		fgadd = mod_internals.fgint;
		fgmult = mnim.calc_gmults(stims);
		sub_outs = fgadd;
		for nn = 1:length(mnim.Msubunits)
			sub_outs(:,mnim.Mtargets(nn)) = fgadd(:,mnim.Mtargets(nn)) .* fgmult(nn);
		end
	end

	%%
	function mnim_out = fit_alt_filters( mnim, Robs, stims, varargin )
	% Usage: mnim_out = fit_alt_filters( mnim, Robs, stims, varargin )
		
		LLtol = 0.0002; MAXiter = 12;

		% Check to see if silent (for alt function)
		silent = 0;
		if ~isempty(varargin)
			for j = 1:length(varargin)
				if strcmp( varargin{j}, 'silent' )
					silent = varargin{j+1};
				end
			end	
		end
					
		varargin{end+1} = 'silent';
		varargin{end+1} = 1;

		LL = mnim.nim.fit_props.LL; LLpast = -1e10;
		if ~silent
			fprintf( 'Beginning LL = %f\n', LL )
		end
		iter = 1;
		mnim_out = mnim;
		while (((LL-LLpast) > LLtol) && (iter < MAXiter))
	
			mnim_out = mnim_out.fit_Mfilters( Robs, stims, varargin );
			mnim_out = mnim_out.fit_filters( Robs, stims, varargin );

			LLpast = LL;
			LL = mnim_out.nim.fit_props.LL;
			iter = iter + 1;

			if ~silent
				fprintf( '  Iter %2d: LL = %f\n', iter, LL )
			end	
		end
	end
		
	%%
	function mnim_out = fit_filters( mnim, Robs, stims, varargin )
	% Usage: mnim = mnim.fit_filters( Robs, stims, Uindx, varargin )
	
		if ~isempty(varargin) && iscell(varargin) && iscell(varargin{1})
			varargin = varargin{1};
		end
		
		varargin{end+1} = 'gain_funs';
		varargin{end+1} = mnim.calc_gmults( stims );
		%varargin{end+1} = 'fit_offsets';
		%varargin{end+1} = 1;
		
		mnim_out = mnim;
		mnim_out.nim = mnim.nim.fit_filters( Robs, stims, varargin );
	end
	
	%%
	function mnim_out = fit_Mfilters( mnim, Robs, stims, varargin )
	% Usage: mnim = mnim.fit_Mfilters( Robs, stims, Uindx, varargin )
	%
	% Enter Msubunits to optimize using 'subs' option, numbered by their index in Msubunits
	
		if ~isempty(varargin) && iscell(varargin) && iscell(varargin{1})
			varargin = varargin{1};
		end
		
		% Determine valid targets
		pos = 1; jj = 1;
		NMsubs = length(mnim.Msubunits);
		Msubs = 1:NMsubs;
		modvarargin = {};
		if ~ischar(varargin{jj})  %if not a flag, it must be train_inds
			modvarargin{pos} = varargin{jj};
      jj = jj + 1; pos = pos + 1;
		end
		while jj <= length(varargin)
			switch varargin{jj}
				case 'subs'
					Msubs = varargin{jj+1};
					jj = jj + 2;
					assert( sum(Msubs > length(mnim.Msubunits)) == 0, 'Invalid multiplicative targets.' )
				otherwise
					modvarargin{pos} = varargin{jj};
					pos = pos + 1;
					jj = jj + 1;
			end
		end
		
		% Extract relevant multiplicative elements for Msubunits
		[subouts,fadd] = mnim.subunit_outputs( stims );
		
		gmults = ones(size(subouts,1),NMsubs+1);
		for nn = 1:Msubs
			gmults(:,nn) = fadd(:,mnim.Mtargets(Msubs(nn)));
		end
		SumXtar = length(stims)+1; 
		% Add all non-targets with additive components from targets
		stims{SumXtar} = zeros(size(subouts,1),1);
		for nn = 1:length(mnim.nim.subunits)
			stims{SumXtar} = stims{SumXtar} + fadd(:,nn);
		end
		
		% Construct NIM for filter minimization
		nimtmp = mnim.nim;
		nimtmp.subunits = mnim.Msubunits(Msubs);
		%for nn = 1:length(Msubs)
		nimtmp.subunits(NMsubs+1) = nimtmp.subunits(NMsubs);
		nimtmp.subunits(end).Xtarg = SumXtar;
		nimtmp.subunits(end).filtK = 1;
		if isfield(nimtmp.subunits(end),'kt')
			nimtmp.subunits(end).kt = 1;
		end
		nimtmp.subunits(end).NLtype = 'lin';
		nimtmp.subunits(end).weight = 1;
		nimtmp.subunits(end).reg_lambdas = SUBUNIT.init_reg_lamdas();
		%end

		% Add Xmatrix with summed non-target components
		stimpar1 = nimtmp.stim_params(1);
		stimpar1.dims = [1 1 1];
		stimpar1.tent_spacing = [];
		nimtmp.stim_params(SumXtar) = stimpar1;
		
		modvarargin{pos} = 'gain_funs';
		modvarargin{pos+1} = gmults;
		modvarargin{pos+2} = 'subs';
		modvarargin{pos+3} = 1:NMsubs;
		%modvarargin{pos+4} = 'fit_offsets';
		%modvarargin{pos+5} = 1;

		nimtmp = nimtmp.fit_filters( Robs, stims, modvarargin );
		
		% Copy filters back to their locations
		mnim_out = mnim;
		mnim_out.Msubunits = nimtmp.subunits(1:NMsubs);
		mnim_out.nim = nimtmp;
		mnim_out.nim.subunits = mnim.nim.subunits;
	end
	
	%%
	function [LL, pred_rate, mod_internals, LL_data] = eval_model( mnim, Robs, stims, varargin )
	%	Usage: [LL, pred_rate, mod_internals, LL_data] = eval_model( mnim, Robs, stims, varargin )

		if ~isempty(varargin)
			if iscell(varargin{1}) && (length(varargin{1}) == 1)
				varargin = varargin{1};
			end	
		end
		
		varargin{end+1} = 'gain_funs';
		gmults = mnim.calc_gmults( stims );
		varargin{end+1} = gmults;

		[LL,pred_rate,mod_internals,LL_data] = mnim.nim.eval_model( Robs, stims, varargin );
		mod_internals.gain_funs = gmults;
	end
	
	%%
	function Msub = clone_to_Msubunit( mnim, subN, weight, filter_flip )
	% Usage: Msub = mnim.clone_to_Msubunit( subN, <weight>, <filter_flip> )
		
		Nmods = length(mnim.nim.subunits);
		assert( subN <= Nmods, 'Invalid subunit to clone.' )
			
		Msub = mnim.nim.subunits(subN);
		if (nargin > 2) && ~isempty(weight)
			Msub.weight = weight;
		end
		if nargin > 3
			Msub.filtK = -Msub.filtK;
			if isfield(Msub,'kt')
				Msub.kt = -Msub.kt;
			end
		end
		
		% Should reset nonlinearity if parametric or nonparametric and normalize filter (output <1)
		if strcmp( Msub.NLtype, 'nonpar' )
			TBy(:) = 0;
		end		
	end
	
	function display_model( mnim, Xstim, Robs )
	% Usage: display_model( mnim, Xstim, Robs )
		
		if nargin < 2
			Xstim = [];
		end
		if nargin < 3
			Robs = [];
		end
		
		nimtmp = mnim.nim;
		Nmods = length(nimtmp.subunits);
		fprintf( 'Regular Subunits: 1-%d\n   Mult Subunits: %d-%d\n', Nmods, Nmods+1, Nmods+length(mnim.Msubunits) )
		nimtmp.subunits = cat(1,nimtmp.subunits, mnim.Msubunits );
		nimtmp.display_model_dab( Xstim, Robs );
	end
	
end

methods (Hidden)
	
	function gmults = calc_gmults( mnim, stims )
	% Usage: gmults = mnim.calc_gmults( stims )
			
		gmults = ones( size(stims{1},1), length(mnim.nim.subunits) );
	
		% Calculate Mfilters
		if ~isempty(mnim.Msubunits)
			nimtmp = mnim.nim;
			nimtmp.subunits = mnim.Msubunits;
			[~,~,mod_int] = nimtmp.eval_model( zeros(size(stims{1},1),1), stims );
			for nn = 1:length(mnim.Mtargets)
				gmults(:,mnim.Mtargets(nn)) = 1 + mnim.Msubunits(nn).weight*mod_int.fgint(:,nn);
			end
		end
	end
	
end
end
