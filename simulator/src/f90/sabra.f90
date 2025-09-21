PROGRAM sabra

    USE params
    USE vars
    USE phys
    USE frc
    USE fmt

    IMPLICIT NONE

    INTEGER :: stat

    INTEGER :: ini
    INTEGER(KIND=8) :: step  ! Uses 64-bit integer
    DOUBLE PRECISION :: dt
    DOUBLE PRECISION :: beta

    INTEGER(KIND=8) :: timet, tstep, tind
    INTEGER(KIND=8) :: timec, cstep
    INTEGER(KIND=8) :: times, sstep, sind
    INTEGER(KIND=8) :: timef, fstep, find
    INTEGER(KIND=8) :: t, i, j
  

    INTEGER :: omax

    DOUBLE COMPLEX, ALLOCATABLE, DIMENSION(:) :: uu
    DOUBLE COMPLEX, ALLOCATABLE, DIMENSION(:) :: ff

    DOUBLE COMPLEX, ALLOCATABLE, DIMENSION(:) :: zz, ww, ss


    DOUBLE PRECISION :: time

    DOUBLE PRECISION :: ener, enst, diss

    DOUBLE PRECISION :: temp, tmp

    DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: spectr

    ! ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----  |

    ! here we create groups of parameters 

    NAMELIST /PARAM/ nn, nu
    NAMELIST /STATU/ stat
    NAMELIST /INOUT/ tstep, cstep, sstep, fstep, omax
    NAMELIST /ALGOP/ step, beta
    NAMELIST /FORCE/ dwn, upp, eps
    NAMELIST /INITU/ e0

    ! here we open the parameters file and for every group of params we read the values

    OPEN(1,file='parameters',status='old')
    READ(1,NML=PARAM)
    READ(1,NML=STATU)
    READ(1,NML=INOUT)
    READ(1,NML=ALGOP)
    READ(1,NML=FORCE)
    READ(1,NML=INITU)
    CLOSE(1)

    ALLOCATE( kk(nn), k2(nn) )
    ALLOCATE( uu(nn), ff(nn) )
    ALLOCATE( zz(nn), ww(nn) )
    ALLOCATE( ss(nn) )

    ALLOCATE( spectr(omax, nn) )

    ! ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----  |

    DO i = 1, nn
        temp = LAM**i   !scale the wavevector of a factor LAM, creating the logscale

        kk(i) = temp
        k2(i) = temp**2
    END DO

    DO i = 1, nn
        ff(i) = 0.0D0 
    END DO

    IF (stat .EQ. 0) THEN

        ini = 1

        CALL initial(uu)

        time = 0.0D0

        tind = 0
        sind = 0
        find = 0

        timet = tstep
        timec = cstep
        times = sstep
        timef = fstep

    ELSE

        ini  = (stat-1)*tstep+1   ! Starting timestep 
        tind = stat               ! Time index for output files
        sind = ini/sstep+1        ! Spectra output index
        find = ini/fstep+1        ! Flux output index
        
        timet = 0
        times = modulo(ini-1,sstep)
        timec = modulo(ini-1,cstep)
        timef = modulo(ini-1,fstep)

        OPEN(1,file='Status/ftime.txt',status='old')
            READ(1,FMT = '(E13.6)') time
        CLOSE(1)
        
        WRITE(ext, fmtext) tind

        OPEN(1,file='Status/uu.'//ext//'.out')
        READ(1,*) uu
        CLOSE(1)
        
    END IF

    10 FORMAT( E13.6,E26.18,E26.18,E26.18 )

    dt = beta * SQRT(nu / eps)

    PRINT*,"INITIALIZATION DONE"

    ! ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----  |

    DO t = ini, step

        CALL forcing(uu, ff)

        ! here we update the status folder every tstep numebr of steps

        IF ( timet.eq.tstep ) THEN
            timet = 0

            OPEN(1,file='Status/ftime.txt')
            WRITE(1,FMT = '(E13.6)') time   !updates final time at which simulation stops
            CLOSE(1)

            tind = tind+1                  ! updates stat
            WRITE(ext, fmtext) tind

            OPEN(1,file='Status/times.txt',position='append')
            WRITE(1,FMT = '(A,E13.6)') ext,time                ! updates times
            CLOSE(1)

            OPEN(1,file='Status/uu.'//ext//'.out')
            WRITE(1,*) uu
            CLOSE(1)

        END IF

        ! ---- ---- ---- ---- ---- ---- ---- ---- ---- ----  |
        ! in this section we update the checks quantities every cstep number of steps. We record in order
        ! time, energy, nu*enstrophy/2 , dissipation

        IF ( timec.eq.cstep ) THEN    
            timec = 0

            ener = 0.0D0
            enst = 0.0D0    ! enstrophy

            DO i = 1, nn
                temp = abs(uu(i))**2

                ener = ener + temp          ! sum of all squared modes of velocities i.e. energy
                enst = enst + temp * k2(i)
            END DO
            
            diss = 0.0D0
            DO i = 1, nn
                diss = diss + REAL( uu(i) * CONJG(ff(i)) ) 
            END DO

            OPEN(1,file='Checks/energy.txt',position='append')
            WRITE(1,10) time, ener, nu*enst/2, diss
            CLOSE(1)

        END IF

        ! ---- ---- ---- ---- ---- ---- ---- ---- ---- ----  |
        ! here we create or modify the spectra files

        IF ( times.eq.sstep ) THEN
            times = 0

            sind = sind+1
            WRITE(ext, fmtext) sind  !the variable ext becomes 00000sind

            DO i = 1, nn
                DO j = 1, omax
                    spectr(j,i) = 0.0D0
                END DO
            END DO
            
            DO i = 1, nn

                tmp = 1.0D0
                temp = abs(uu(i))

                DO j = 1, omax             ! Loop over moments
                    tmp = tmp * temp       ! tmp = |u_n|^j so that we can compute structure functions up to power omax
                    spectr(j,i) = spectr(j,i) + tmp  ! Add j-th moment to spectrum
                END DO
            END DO

            OPEN(1,file='Spectra/spectr.' // ext // '.txt',position='append')
            DO i = 1, nn
                WRITE(1,'(E13.6)' ,advance='no')  kk(i)
                DO j = 1, omax
                    WRITE(1,'(ES26.18E3)',advance='no')  spectr(j,i)
                END DO
                WRITE(1,*)
            ENDDO
            CLOSE(1)

        END IF

        ! ---- ---- ---- ---- ---- ---- ---- ---- ---- ----  |

        IF ( timef.eq.fstep ) THEN
            timef = 0

            find = find+1
            WRITE(ext, fmtext) find

            CALL eflux(uu, ext)

            OPEN(1,file='Results/times.txt',position='append')
            WRITE(1,FMT = '(A,E13.6)') ext,time
            CLOSE(1)

        END IF

        ! --- > | STEP 1 | < ------------------------------------ |

        CALL nonl(uu, zz)

        DO i = 1, nn

            zz(i) = - nu * k2(i) * uu(i) + im * zz(i) + ff(i)  !add forcing and linear term to the nonlinear one
            
            ss(i) = uu(i) + dt * zz(i)/6
            ww(i) = uu(i) + dt * zz(i)/2
                    
        END DO
        
        ! --- > | STEP 2 | < ------------------------------------ |

        CALL forcing(ww, ff)
        CALL nonl(ww, zz)

        DO i = 1, nn

            zz(i) = - nu * k2(i) * ww(i) + im * zz(i) + ff(i)
            
            ss(i) = ss(i) + dt * zz(i)/3
            ww(i) = uu(i) + dt * zz(i)/2
                    
        END DO
        
        ! --- > | STEP 3 | < ------------------------------------ |

        CALL forcing(ww, ff)
        CALL nonl(ww, zz)

        DO i = 1, nn

            zz(i) = - nu * k2(i) * ww(i) + im * zz(i) + ff(i)
            
            ss(i) = ss(i) + dt * zz(i)/3
            ww(i) = uu(i) + dt * zz(i)
                    
        END DO

        ! --- > | STEP 4 | < ------------------------------------ |

        CALL forcing(ww, ff)
        CALL nonl(ww, zz)

        DO i = 1, nn

            zz(i) = - nu * k2(i) * ww(i) + im * zz(i) + ff(i)
            
            uu(i) = ss(i) + dt * zz(i)/6
                    
        END DO

        timet = timet + 1
        timec = timec + 1
        times = times + 1
        timef = timef + 1
        time  = time  + dt

    END DO

    DEALLOCATE( uu, ff )
    DEALLOCATE( kk )

    PRINT*,"END"

END PROGRAM sabra