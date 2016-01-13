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
%  methods 
		%[] = display_model(nim,Robs,Xstims,varargin); %display current model
%	end
	
%	methods (Static)
    %Xmat = create_time_embedding( stim, params ) %make time-embedded stimulus
%	end
%	methods (Static, Hidden)
		%Tmat = create_Tikhonov_matrix( stim_params, reg_type ); %make regularization matrices
%	end
	
	
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
		if nargin < 2
			return
		end
		assert( nargin == 3, 'Must specify targets as well as subunits' )
		assert( length(Msubunits) == length(Mtargets), 'Mtargets out of range.' )
		mnim.Msubunits = Msubunits;
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
			assert( subunit <= length(mnim.nim.subunits), 'subunit to be multiplicative must be a normal subunit' )
			Msubunit = mnim.subunits(subunit);
			% Default nonlinearity/scaling?
		else
			mnim.Msubunits = cat(1, mnim.Msubunits, subunit );
		end
	
		mnim.Mtargets(end+1) = Mtarget;
	end
	
	%%
	function mnim_out = fit_alt_filters( mnim, Robs, stims, varargin )
	% Usage: mnim_out = fit_alt_filters( mnim, Robs, stims, varargin )
		
		LLtol = 0.0002; MAXiter = 12;

		% Check to see if silent (for alt function)
		[~,parsed_options,modvarargin] = NIM.parse_varargin( varargin, {'silent'} );
		if isfield( parsed_options, 'silent' )
			silent = parsed_options.silent;
		else
			silent = 0;
		end
					
		modvarargin{end+1} = 'silent';
		modvarargin{end+1} = 1;

		LL = mnim.nim.fit_props.LL; LLpast = -1e10;
		if ~silent
			fprintf( 'Beginning LL = %f\n', LL )
		end
		iter = 1;
		mnim_out = mnim;
		while (((LL-LLpast) > LLtol) && (iter < MAXiter))
	
			mnim_out = mnim_out.fit_Mfilters( Robs, stims, modvarargin{:} );
			mnim_out = mnim_out.fit_filters( Robs, stims, modvarargin{:} );

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
	
		if ~iscell(stims)
			tmp = stims;
			clear stims
			stims{1} = tmp;
		end
		
		varargin{end+1} = 'gain_funs';
		varargin{end+1} = mnim.calc_gmults( stims );
		%varargin{end+1} = 'fit_offsets';
		%varargin{end+1} = 1;
		
		mnim_out = mnim;
		mnim_out.nim = mnim.nim.fit_filters( Robs, stims, varargin{:} );
	end
	
	%%
	function mnim_out = fit_Mfilters( mnim, Robs, stims, varargin )
	% Usage: mnim = mnim.fit_Mfilters( Robs, stims, Uindx, varargin )
	%
	% Enter Msubunits to optimize using 'subs' option, numbered by their index in Msubunits
	
		if ~iscell(stims)
			tmp = stims;
			clear stims
			stims{1} = tmp;
		end

		[~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs'} );
		if isfield( parsed_inputs, 'subs' )
			Mtar = parsed_inputs.subs;
		else
			Mtar = 1:length( mnim.Msubunits );
		end
		
		NMsubs = length(Mtar);
		[nimtmp,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtar );

		modvarargin{end+1} = 'gain_funs';
		modvarargin{end+1} = gmults;
		modvarargin{end+1} = 'subs';
		modvarargin{end+1} = 1:NMsubs;
		%modvarargin{pos+4} = 'fit_offsets';
		%modvarargin{pos+5} = 1;

		nimtmp = nimtmp.fit_filters( Robs, stims_plus, modvarargin{:} );
		
		% Copy filters back to their locations
		mnim_out = mnim;
		mnim_out.Msubunits = nimtmp.subunits(1:NMsubs);
		mnim_out.nim = nimtmp;
		mnim_out.nim.subunits = mnim.nim.subunits;
	end

	%%
	function mnim_out = fit_upstreamNLs( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_upstreamNLs( Robs, stims, Uindx, varargin )
	
		if ~iscell(stims)
			tmp = stims;
			clear stims
			stims{1} = tmp;
		end
		
		varargin{end+1} = 'gain_funs';
		varargin{end+1} = mnim.calc_gmults( stims );
		%varargin{end+1} = 'fit_offsets';
		%varargin{end+1} = 1;
		
		mnim_out = mnim;
		mnim_out.nim = mnim.nim.fit_upstreamNLs( Robs, stims, varargin{:} );
	end

	function mnim_out = fit_MupstreamNLs( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_MupstreamNLs( Robs, stims, Uindx, varargin )
	
		if ~iscell(stims)
			tmp = stims;
			clear stims
			stims{1} = tmp;
		end
		
		[~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs'} );
		if isfield( parsed_inputs, 'subs' )
			Mtar = parsed_inputs.subs;
		else
			Mtar = 1:length( mnim.Msubunits );
		end
		
		NMsubs = length(Mtar);
		[nimtmp,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtar );

		modvarargin{end+1} = 'gain_funs';
		modvarargin{end+1} = gmults;
		modvarargin{end+1} = 'subs';
		modvarargin{end+1} = 1:NMsubs;
		%modvarargin{pos+4} = 'fit_offsets';
		%modvarargin{pos+5} = 1;

		nimtmp = nimtmp.fit_upstreamNLs( Robs, stims_plus, modvarargin{:} );
		
		% Copy filters back to their locations
		mnim_out = mnim;
		mnim_out.Msubunits = nimtmp.subunits(1:NMsubs);
		mnim_out.nim = nimtmp;
		mnim_out.nim.subunits = mnim.nim.subunits;
	end

	
	
	%%
	function mnim = reg_path( mnim, Robs, stims, Uindx, XVindx, varargin )
	%	Usage: mnim = reg_path( mnim, Robs, stims, Uindx, XVindx, varargin )

		if ~iscell(stims)
			tmp = stims;
			clear stims
			stims{1} = tmp;
		end
		
		varargin{end+1} = 'gain_funs';
		varargin{end+1} = mnim.calc_gmults( stims );
		%varargin{end+1} = 'fit_offsets';
		%varargin{end+1} = 1;
		
		mnim_out = mnim;
		mnim_out.nim = mnim.nim.reg_path( Robs, stims, Uindx, XVindx, varargin{:} );
	end
	
	%%
	function mnim = reg_pathM( mnim, Robs, stims, Uindx, XVindx, varargin )
	%	Usage: mnim = reg_pathM( mnim, Robs, stims, Uindx, XVindx, varargin )

		[~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs'} );
		if isfield( parsed_inputs, 'subs' )
			Mtar = parsed_inputs.subs;
		else
			Mtar = 1:length( mnim.Msubunits );
		end

		[nimtmp,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtar );
		
		modvarargin{end+1} = 'gain_funs';
		modvarargin{end+1} = gmults;
		modvarargin{end+1} = 'subs';
		modvarargin{end+1} = Mtar;
		
		mnim_out = mnim;
		mnim_out.nim = nimtmp.reg_path( Robs, stims_plus, Uindx, XVindx, modvarargin{:} );
	end
	
	%%
	function [LL, pred_rate, mod_internals, LL_data] = eval_model( mnim, Robs, stims, varargin )
	%	Usage: [LL, pred_rate, mod_internals, LL_data] = eval_model( mnim, Robs, stims, varargin )
		
		varargin{end+1} = 'gain_funs';
		gmults = mnim.calc_gmults( stims );
		varargin{end+1} = gmults;

		[LL,pred_rate,mod_internals,LL_data] = mnim.nim.eval_model( Robs, stims, varargin{:} );
		mod_internals.gain_funs = gmults;
	end
	
	%% 
	function mnim = init_nonpar_NLs( mnim, stims, varargin )
	% Usage: mnim = mnim.init_nonpar_NLs( stims, varargin )
	%
	%  Initializes the specified model subunits to have nonparametric (tent-basis) upstream NLs,
	%  inherited from NIM version. 
	%
	%  Note: default is initializes all subunits and Msubunits. Change this through optional flags
	%
	%  INPUTS: 
	%			stims: cell array of stimuli
	%     optional flags:
	%        ('subs',sub_inds): Index values of set of subunits to make nonpar (default is all)
	%        ('Msubs',sub_inds): Index values of set of Msubunits to make nonpar (default is all)
	%        ('lambda_nld2',lambda_nld2): specify strength of smoothness regularization for the tent-basis coefs
	%        ('NLmon',NLmon): Set to +1 to constrain NL coefs to be monotonic increasing and
	%						 -1 to make monotonic decreasing. 0 means no constraint. Default here is +1 (monotonic increasing)
	%				 ('edge_p',edge_p): Scalar that determines the locations of the outermost tent-bases 
	%            relative to the underlying generating distribution
	%        ('n_bfs',n_bfs): Number of tent-basis functions to use 
	%        ('space_type',space_type): Use either 'equispace' for uniform bin spacing, or 'equipop' for 'equipopulated bins' 
	%  	OUTPUTS: mnim: new mnim object

		
		% Initialize subunit NLs
		[~,parsed_options,modvarargin] = NIM.parse_varargin( varargin, {'Msubs'} );
		mnim.nim = mnim.nim.init_nonpar_NLs( stims, modvarargin{:} );
		
		% Initialize Msubunit NLs
		if isfield(parsed_options,'Msubs')
			Msubs = parsed_options.Msubs;
		else
			Msubs = 1:length(mnim.Msubunits);
		end
		if ~isempty(Msubs)
			[~,~,modvarargin] = NIM.parse_varargin( modvarargin, {'subs'} );
			modvarargin{end+1} = 'subs';
			modvarargin{end+1} = Msubs;
			nimtmp = mnim.nim;
			nimtmp.subunits = mnim.Msubunits;
			nimtmp = nimtmp.init_nonpar_NLs( stims, modvarargin{:} );
			mnim.Msubunits = nimtmp.subunits;
		end	
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
		if strcmp(class(nimtmp),'NIM')
			nimtmp.display_model_dab( Xstim, Robs );
		else
			nimtmp.display_model( Xstim, Robs );
		end
	end

	%%
	function [sub_outs,fgadd,fgmult] = subunit_outputs( mnim, stims )
	% Usage: [sub_outs,fgadd,fgmult] = subunit_outputs( mnim, stims )
	% 
	% Calculate output of all subunits (addxmult) and separate additive and multiplicative
	
		[~,~,mod_internals] = mnim.nim.eval_model( [], stims );
		fgadd = mod_internals.fgint;
		
		% multiply by excitatory and inhibitory weights
		for nn = 1:length(mnim.nim.subunits)
			fgadd(:,nn) = fgadd(:,nn) * mnim.nim.subunits(nn).weight;
		end
		
		% calculate multiplicative effect
		fgmult = mnim.calc_gmults(stims);
		sub_outs = fgadd;
		for nn = 1:length(mnim.Msubunits)
			sub_outs(:,mnim.Mtargets(nn)) = fgadd(:,mnim.Mtargets(nn)) .* fgmult(mnim.Mtargets(nn));
		end
	end

	
end

%%
methods (Hidden)
	
	function gmults = calc_gmults( mnim, stims )
	% Usage: gmults = mnim.calc_gmults( stims )
			
		%gmults = ones( size(stims{1},1), length(mnim.nim.subunits) );
		[~,~,mod_int] = mnim.nim.eval_model( [], stims ); % cheap way to get size of gmults for diff nim classes
		gmults = ones(size(mod_int.gint));
	
		% Calculate Mfilters
		if ~isempty(mnim.Msubunits)
			nimtmp = mnim.nim;
			nimtmp.subunits = mnim.Msubunits;
			[~,~,mod_int] = nimtmp.eval_model( [], stims );
			for nn = 1:length(mnim.Mtargets)
				gmults(:,mnim.Mtargets(nn)) = 1 + mnim.Msubunits(nn).weight*mod_int.fgint(:,nn);
			end
		end
	end
	
	
	%%
	function [nim,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtargets )
	% Usage: [nim,gmults,stims] = format4Mfitting( mnim, stims, targets )
	% 
	% set up for fitting multiplicative subunits as NIM with gmults and addition stim-matrices
	
		if ~iscell(stims)
			stims_plus{1} = stims;
		else
			stims_plus = stims;
		end
		NMsubs = length(Mtargets);

		% Extract relevant multiplicative elements for Msubunits
		[subouts,fadd] = mnim.subunit_outputs( stims_plus );
		
		% Assign subunits as multipliers for targets
		gmults = ones(size(subouts,1),NMsubs+1);  % extra-dim of ones is to multiply additive term
		for nn = 1:NMsubs
			gmults(:,nn) = fadd(:,mnim.Mtargets(Mtargets(nn)));
		end
		SumXtar = length(stims_plus)+1; 
		
		% Add all non-targets with additive components from targets
		stims_plus{SumXtar} = zeros(size(subouts,1),1);
		for nn = 1:length(mnim.nim.subunits)
			if ismember(nn,Mtargets)
				stims_plus{SumXtar} = stims_plus{SumXtar} + fadd(:,nn); % additive component if target
			else
				stims_plus{SumXtar} = stims_plus{SumXtar} + subouts(:,nn); % fully multiplied component of nontargets
			end
		end
		
		% Construct NIM for filter minimization
		nim = mnim.nim;
		nim.subunits = mnim.Msubunits(Mtargets);
		%for nn = 1:length(Msubs)
		
		% make last NIM-subunit add in additive terms
		nim.subunits(NMsubs+1) = nim.subunits(NMsubs);
		nim.subunits(end).Xtarg = SumXtar;
		nim.subunits(end).filtK = 1;
		nim.subunits(end).NLoffset = 0;
		if isa(nim.subunits(end),'LRSUBUNIT')
			nim.subunits(end).kt = 1;
			nim.subunits(end).ksp = 1;
		end
		nim.subunits(end).NLtype = 'lin';
		nim.subunits(end).weight = 1;
		nim.subunits(end).reg_lambdas = SUBUNIT.init_reg_lamdas();
		%end

		% Add Xmatrix with summed non-target components
		stimpar1 = nim.stim_params(1);
		stimpar1.dims = [1 1 1];
		stimpar1.tent_spacing = [];
		stimpar1.up_fac = 1;  % since any up_fac is already taken into account in subunit outputs
		nim.stim_params(SumXtar) = stimpar1;
				
	end
	
end
end
